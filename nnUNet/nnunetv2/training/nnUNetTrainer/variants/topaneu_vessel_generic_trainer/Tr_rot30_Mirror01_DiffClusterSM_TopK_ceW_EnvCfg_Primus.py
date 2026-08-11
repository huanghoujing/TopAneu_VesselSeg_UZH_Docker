"""
Primus (transformer) trainer that reuses ALL the tricks of the EnvCfg CNN trainer
(Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg): the DC_SkelREC_Diff_ClusterSM_and_CE_TopK loss
(ce_class_weight, TopK, Dice, skeleton-recall, Diff, ClusterSM), the skel dataloaders incl.
cls_balanced_global_sampling, the rot30/Mirror augmentation, env-driven config, the every-N-epoch
actual validation, and the custom run_training loop.

What this trainer changes vs the CNN parent:
- network: a Primus / PrimusV2* / PrimusV3* transformer (no deep supervision)
- optimizer: AdamW + linear warmup -> poly decay (ported from official AbstractPrimus / nnUNetTrainer_warmup)
- gradient clipping to norm 1 (Primus), not 12
- ClusterSM feature tap: PatchDecode_wF returns {'seg_outputs', 'feature'} like UNetDecoder_wF, with an
  optional CLI-configurable full-res head (PRIMUS_HEAD_CONVS / PRIMUS_HEAD_WIDTH). Without it, the
  feature is the native half-res pre-classifier tensor and ClusterSoftmaxLoss downsamples the target.

Standalone inference works without predictor edits: the full Primus spec (variant, patch_size, head
config) is injected into configuration_manager.network_arch_init_kwargs at __init__, which is saved
into the run's plans.json and read back by the static build_network_architecture (B3b in the plan).

Env vars (all optional):
  PRIMUS_MODEL        S|B|M|L|V2S|V2B|V2M|V2L|V3S|V3B|V3M|V3L   (default M)
  PRIMUS_HEAD_CONVS   int, number of full-res conv blocks in the head (default 0 -> half-res feature)
  PRIMUS_HEAD_WIDTH   int, channels of the full-res head (default channels[-2])
  warmup_epochs       linear-warmup duration (default 50)
  initial_lr          AdamW peak lr (also overridable via --initial_lr CLI; default 3e-4)
  weight_decay        AdamW weight decay (default 5e-2)
plus everything the EnvCfg parent already reads (n_classes_w_bg, ce_class_weight, weight_*, etc.).
"""
import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule

import dynamic_network_architectures.architectures.primus as primus_mod
from dynamic_network_architectures.architectures.primus import (
    Primus, PrimusV2S, PrimusV2B, PrimusV2M, PrimusV2L,
)
try:
    from dynamic_network_architectures.architectures.primus import (
        PrimusV3S, PrimusV3B, PrimusV3M, PrimusV3L,
    )
except ImportError:
    PrimusV3S = PrimusV3B = PrimusV3M = PrimusV3L = None
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchDecode, LayerNormNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks

from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_warmup import nnUNetTrainer_warmup
from nnunetv2.training.nnUNetTrainer.variants.topaneu_vessel_generic_trainer.Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg import \
    Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg


# Per-size Primus(V1) config: (embed_dim, eva_depth, eva_numheads). Mirrors official nnUNet_Primus_*_Trainer.
_PRIMUS_V1 = {'S': (396, 12, 6), 'B': (792, 12, 12), 'M': (864, 16, 12), 'L': (1056, 24, 16)}
# V2/V3 encode the size in the class itself.
_PRIMUS_VX = {
    'V2S': PrimusV2S, 'V2B': PrimusV2B, 'V2M': PrimusV2M, 'V2L': PrimusV2L,
    'V3S': PrimusV3S, 'V3B': PrimusV3B, 'V3M': PrimusV3M, 'V3L': PrimusV3L,
}


class PatchDecode_wF(PatchDecode):
    """PatchDecode that also exposes the pre-classifier feature for ClusterSM (mirrors UNetDecoder_wF).

    Head config is taken from class attributes set by build_network_architecture (so it is identical
    at train and inference time, read from the persisted plans). With _HEAD_CONVS == 0 we keep stock
    Primus behavior: the final stride-2 ConvTranspose maps the half-res feature straight to logits, so
    the exposed feature is half-res. With _HEAD_CONVS > 0 we upsample to full res, run N conv blocks
    (reusing nnUNet's StackedConvBlocks, parameterized to Primus' LayerNormNd/GELU style) and a 1x1
    classifier, exposing a true full-res feature.
    """
    _HEAD_CONVS = 0
    _HEAD_WIDTH = None
    _DS_HEADS = False   # build deep-supervision aux heads (gated on the persisted _primus spec)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        n = int(type(self)._HEAD_CONVS or 0)
        if n > 0:
            ch_in = self.decode[-1].in_channels          # channels[-2]
            out_ch = self.decode[-1].out_channels        # num_classes
            stride = self.decode[-1].stride              # final upsample stride
            c_head = int(type(self)._HEAD_WIDTH) if type(self)._HEAD_WIDTH else ch_in
            self.upsample = nn.Sequential(               # half-res -> full-res feature
                nn.ConvTranspose3d(ch_in, c_head, kernel_size=stride, stride=stride),
                LayerNormNd(c_head), nn.GELU())
            self.head = StackedConvBlocks(n, nn.Conv3d, c_head, c_head, 3, 1,
                                          norm_op=LayerNormNd, nonlin=nn.GELU)  # full-res blocks
            self.classifier = nn.Conv3d(c_head, out_ch, kernel_size=1)
            self.decode = self.decode[:-1]               # keep stages up to half-res
            self._wF_fullres = True
        else:
            self._wF_fullres = False

        # Deep supervision: 1x1 aux seg heads on the intermediate decode-stage features (mirrors
        # UNetDecoder's per-stage seg_layers). Built only when DS was on (persisted spec) so existing
        # non-DS checkpoints stay loadable. self.deep_supervision is the runtime train/val toggle.
        self.deep_supervision = False
        self.ds_heads = None
        if type(self)._DS_HEADS:
            out_ch = self.classifier.out_channels if self._wF_fullres else self.decode[-1].out_channels
            n_feat = len(self.decode) if self._wF_fullres else (len(self.decode) - 1)
            self.ds_heads = nn.ModuleList([nn.Conv3d(self.decode[s][0].out_channels, out_ch, 1)
                                           for s in range(n_feat)])

    def forward(self, x):
        # walk the upsampling (feature-producing) decode stages, keeping each intermediate feature
        n_feat = len(self.decode) if self._wF_fullres else (len(self.decode) - 1)
        feats, h = [], x
        for s in range(n_feat):
            h = self.decode[s](h)
            feats.append(h)                                   # feats[0]=1/4, feats[1]=1/2, ...
        if self._wF_fullres:
            feat = self.head(self.upsample(feats[-1]))        # full-res hidden feature
            out = self.classifier(feat)                       # full-res logits
        else:
            feat = feats[-1]                                  # half-res pre-classifier feature
            out = self.decode[-1](feat)                       # stride-2 conv -> full-res logits
        if self.training:
            if getattr(self, 'deep_supervision', False) and self.ds_heads is not None:
                # full-res first, then deeper levels (1/2, 1/4, ...) to match DownsampleSegForDSTransform
                seg = [out] + [self.ds_heads[s](feats[s]) for s in range(len(feats))][::-1]
                return {'seg_outputs': seg, 'feature': feat}
            return {'seg_outputs': out, 'feature': feat}
        return out


class Tr_rot30_Mirror01_DiffSM_TopK_ceW_Primus_EnvCfg(Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg,
                                                      nnUNetTrainer_warmup):
    # MRO: this -> EnvCfg -> ceW -> nnUNetTrainer_warmup -> nnUNetTrainer. We override configure_optimizers
    # (AdamW) and on_train_epoch_start below; load_checkpoint is inherited from nnUNetTrainer_warmup
    # (warmup-aware resume) and reuses our configure_optimizers.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # all env-cfg tricks, loss, dataloaders, transforms, validation

        # ---- Primus / warmup hyperparameters ----
        # initial_lr priority: env 'initial_lr' > CLI --initial_lr (if not the base 1e-2 default) > 3e-4
        self.initial_lr = float(os.environ.get('initial_lr',
                                               self.initial_lr if self.initial_lr != 1e-2 else 3e-4))
        self.weight_decay = float(os.environ.get('weight_decay', 5e-2))
        # enable_deep_supervision / num_ds_levels flow from the EnvCfg parent (env-driven). DS taps the
        # PatchDecode intermediate features via PatchDecode_wF aux heads; capped at 2 down-sampled levels.
        self.gradient_clip_norm = 1  # Primus clips to 1 (parent ceW.train_step reads this); default is 12
        # warmup_duration_whole_net ('warmup_epochs') and training_stage come from nnUNetTrainer_warmup
        # class-level defaults (this MRO bypasses warmup.__init__ via EnvCfg's direct nnUNetTrainer.__init__).

        patch_size = self.configuration_manager.patch_size
        assert all(p % 8 == 0 for p in patch_size), \
            f"Primus needs patch_size divisible by 8, got {patch_size}"

        # Persist the full Primus spec so the network can be rebuilt at standalone inference.
        # IMPORTANT: configuration_manager.configuration is a *deepcopy* of plans['configurations'][cfg]
        # (PlansManager.get_configuration -> _internal_resolve_configuration_inheritance), so mutating
        # configuration_manager alone does NOT reach self.plans_manager.plans, which is what
        # on_train_start saves to plans.json. We therefore write the spec to BOTH:
        #   1. configuration_manager  -> used by this run's initialize() to build the network now;
        #   2. plans_manager.plans    -> saved into the run's plans.json -> read back by nnUNetv2_predict.
        # build_network_architecture reads env-FIRST and falls back to this spec, so env still overrides
        # it and official checkpoints that lack '_primus' keep working via env vars.
        primus_spec = {
            'variant': os.environ.get('PRIMUS_MODEL', 'M'),
            'patch_size': [int(p) for p in patch_size],
            'head_convs': int(os.environ.get('PRIMUS_HEAD_CONVS', 0)),
            'head_width': (int(os.environ['PRIMUS_HEAD_WIDTH']) if os.environ.get('PRIMUS_HEAD_WIDTH') else None),
            'deep_supervision': bool(self.enable_deep_supervision),  # gates aux-head creation at build time
        }
        self.configuration_manager.network_arch_init_kwargs['_primus'] = primus_spec
        try:
            self.plans_manager.plans['configurations'][self.configuration_name][
                'architecture']['arch_kwargs']['_primus'] = primus_spec
        except (KeyError, TypeError):
            self.print_to_log_file(
                "[Primus] WARNING: could not persist '_primus' into plans.json (no architecture.arch_kwargs "
                "in raw plans). Standalone inference will require the PRIMUS_* env vars to be set.")

        self.print_to_log_file(
            f"[{self.__class__.__name__}] Primus spec: {self.configuration_manager.network_arch_init_kwargs['_primus']}")
        self.print_to_log_file(
            f"[{self.__class__.__name__}] initial_lr (AdamW): {self.initial_lr}, weight_decay: {self.weight_decay}, "
            f"warmup_duration_whole_net: {self.warmup_duration_whole_net}")

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        """Build the Primus network. Same static signature as the base trainer so both training
        (`initialize()`) and inference (`nnUNetv2_predict`) call it unchanged.

        All four knobs (variant / patch_size / head_convs / head_width) are read ENV-FIRST, falling
        back to the spec persisted in plans.json by __init__ (patch_size env is 'PRIMUS_PATCH_SIZE',
        e.g. '64,128,192'). NOTE: every one of them determines the network's parameter names / shapes,
        so at inference the env vars must either be unset (-> persisted spec is used) or set to the
        SAME values used at training; otherwise the checkpoint will not load."""
        spec = arch_init_kwargs.get('_primus', {})
        if os.environ.get('PRIMUS_PATCH_SIZE'):
            patch_size = tuple(int(p) for p in os.environ['PRIMUS_PATCH_SIZE'].split(','))
        else:
            assert 'patch_size' in spec, \
                "No patch_size: set PRIMUS_PATCH_SIZE (e.g. '64,128,192') or ensure the '_primus' spec " \
                "(injected by the trainer __init__ and persisted into plans.json) is present."
            patch_size = tuple(int(p) for p in spec['patch_size'])
        variant = str(os.environ.get('PRIMUS_MODEL', spec.get('variant', 'M'))).upper()

        # Apply the PatchDecode monkeypatch HERE (not at module import) so that merely importing this
        # file during nnUNet's trainer-folder scan does not globally alter
        # dynamic_network_architectures for other trainers. Primus builds
        # `self.up_projection = PatchDecode(...)` in its base __init__, resolving PatchDecode from the
        # primus module's globals -> our subclass. Covers Primus / PrimusV2* / PrimusV3*. The head
        # config flows to PatchDecode_wF via class attrs, set BEFORE Primus builds up_projection.
        primus_mod.PatchDecode = PatchDecode_wF
        PatchDecode_wF._HEAD_CONVS = int(os.environ.get('PRIMUS_HEAD_CONVS', spec.get('head_convs', 0) or 0))
        PatchDecode_wF._HEAD_WIDTH = (int(os.environ['PRIMUS_HEAD_WIDTH'])
                                      if os.environ.get('PRIMUS_HEAD_WIDTH')
                                      else spec.get('head_width', None))
        # Deep-supervision aux heads exist iff DS was on at train time (persisted spec), env-overridable.
        # Must NOT depend on the enable_deep_supervision arg (False at inference) or the checkpoint would
        # mismatch a DS-trained model.
        PatchDecode_wF._DS_HEADS = (os.environ.get('enable_deep_supervision') == '1') \
                                   or bool(spec.get('deep_supervision', False))

        if variant in _PRIMUS_V1:
            embed_dim, eva_depth, eva_numheads = _PRIMUS_V1[variant]
            model = Primus(
                num_input_channels, embed_dim, (8, 8, 8), num_output_channels, eva_depth, eva_numheads,
                patch_size, drop_path_rate=0.2, scale_attn_inner=True, init_values=0.1)
        elif variant in _PRIMUS_VX:
            cls = _PRIMUS_VX[variant]
            if cls is None:
                raise ValueError(f"Primus variant {variant} unavailable in installed "
                                 f"dynamic-network-architectures (need a version that provides it).")
            model = cls(
                num_input_channels, num_output_channels, patch_embed_size=(8, 8, 8), input_shape=patch_size,
                drop_path_rate=0.2, scale_attn_inner=True, init_values=0.1)
        else:
            raise ValueError(f"Unknown PRIMUS_MODEL '{variant}'. "
                             f"Expected one of {sorted(set(_PRIMUS_V1) | set(_PRIMUS_VX))}.")
        return model

    # ---- AdamW + warmup optimizer (ported from official AbstractPrimus.configure_optimizers) ----
    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]
        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        params = self.network.module.parameters() if isinstance(self.network, DDP) else self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(params, self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                optimizer = self.optimizer  # keep accumulated momentum
            else:
                optimizer = torch.optim.AdamW(params, self.initial_lr, weight_decay=self.weight_decay,
                                              amsgrad=False, betas=(0.9, 0.98), fused=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs,
                                                  self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")
        super().on_train_epoch_start()

    def set_deep_supervision_enabled(self, enabled: bool):
        # Toggle the runtime DS flag on the PatchDecode head (no-op if aux heads were not built).
        mod = self.network
        if isinstance(mod, DDP):
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        if getattr(mod.up_projection, 'ds_heads', None) is not None:
            mod.up_projection.deep_supervision = enabled

    def _get_deep_supervision_scales(self):
        # Primus' PatchDecode upsamples by the (8,8,8) patch-embed => log2(8)=3 stride-2 stages, so DS
        # heads sit at scales 1, 1/2, 1/4 (full-res first). This replaces the CNN scales derived from
        # pool_op_kernel_sizes, which don't match the transformer decoder.
        if not self.enable_deep_supervision:
            return None
        n = 3
        return [[1.0 / (2 ** i)] * 3 for i in range(n)]   # [[1,1,1],[0.5,0.5,0.5],[0.25,0.25,0.25]]

    # train_step is inherited from the ceW parent (skel-passing). It clips gradients to
    # self.gradient_clip_norm, which we set to 1 in __init__ (parent default 12).
