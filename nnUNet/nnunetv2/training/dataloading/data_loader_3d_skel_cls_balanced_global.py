import numpy as np

from nnunetv2.training.dataloading.data_loader_3d_skel import nnUNetDataLoader3DSkel


def case_sampling_weight(key, weight_groups):
    """Return the sampling weight ("potential") for a case key.

    weight_groups is an ordered list of (patterns, weight) tuples, where patterns is a list of substrings. The
    weight of the FIRST group with any pattern occurring in key is returned; if no group matches, the default
    weight 1.0 is returned. This is used to down- (or up-) weight cases by naming pattern, e.g. cross-modality
    registration generated cases whose names contain "CT2MR"/"MR2CT".
    """
    if weight_groups:
        for patterns, w in weight_groups:
            if any(p in key for p in patterns):
                return float(w)
    return 1.0


class nnUNetDataLoader3DSkelClsBalancedGlobal(nnUNetDataLoader3DSkel):
    """
    Like nnUNetDataLoader3DSkel, but when a sample is forced to be foreground the foreground class for the patch
    center is drawn FIRST (from a configurable, by-default-uniform distribution over all foreground labels), and
    then a case that actually contains that class is selected. This makes the patch center globally balanced across
    foreground classes instead of nnUNet's default (case first, then a class present in that case), which
    under-samples classes that only appear in a few cases.

    Whether a sample is forced foreground is decided per-sample by ``np.random.uniform() < oversample_foreground_percent``
    (pass ``probabilistic_oversampling=True`` when constructing this loader).

    The per-class sampling weights can be set in three mutually exclusive ways:
      1. uniform (default): every fg class is equally likely.
      2. manual: pass ``fg_class_sampling_weights`` aligned to ``fg_labels``.
      3. calibrated: pass ``fg_class_calibration_degree`` (a float). The baseline (degree 0) is each class's
         natural ratio in the dataset, measured as case frequency (fraction of training cases containing the
         class); ``prob_c proportional to f_c ** (1 - degree)``. degree 0 -> natural ratio, degree 1 -> uniform,
         degree > 1 -> over-samples rare classes (rare = present in few cases).

    Args (in addition to the base loader):
        fg_labels: ordered list of all foreground label integers (e.g. label_manager.foreground_labels). The
            weights below are aligned to this order. If None, inferred as the sorted integer keys found in the
            cases' class_locations.
        fg_class_sampling_weights: per-class sampling weights aligned to ``fg_labels`` (need not sum to 1; they are
            normalized). If None, weights come from ``fg_class_calibration_degree`` (if set) or are uniform.
        fg_class_calibration_degree: see mode 3 above. Ignored if ``fg_class_sampling_weights`` is given.
            Classes that are absent from every case, or whose final weight is <= 0, are dropped (with a warning)
            and the remaining weights are renormalized.
        case_sampling_weight_groups: optional ordered list of (patterns, weight) tuples used to weight HOW OFTEN
            each case is chosen by name (see ``case_sampling_weight``). It affects the foreground path here
            (within the chosen fg class, a case is drawn proportional to its weight instead of uniformly). The
            background path is weighted separately via ``sampling_probabilities`` (set by the trainer from the
            same groups). If None, cases are chosen uniformly (weight 1 each).
        print_to_log_file: logging callable (e.g. the trainer's print_to_log_file) used for the startup index
            report. Defaults to the builtin print. Only used during __init__ (in the main process) and never
            stored on the instance, so the loader stays picklable for the augmenter worker processes.
    """

    def __init__(self, *args, fg_labels=None, fg_class_sampling_weights=None, fg_class_calibration_degree=None,
                 case_sampling_weight_groups=None, print_to_log_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_class_index(fg_labels, fg_class_sampling_weights, fg_class_calibration_degree,
                                case_sampling_weight_groups,
                                print_to_log_file if print_to_log_file is not None else print)

    def _build_class_index(self, fg_labels, fg_class_sampling_weights, fg_class_calibration_degree,
                           case_sampling_weight_groups, print_to_log_file):
        # map each integer fg label -> list of case keys that contain at least one voxel of that label
        class_to_keys = {}
        for k in self._data.keys():
            class_locations = self._data[k]['properties']['class_locations']
            for c, locs in class_locations.items():
                # skip the annotated_classes_key tuple (ignore-label bookkeeping); we only want plain integer labels
                if isinstance(c, (int, np.integer)) and locs is not None and len(locs) > 0:
                    class_to_keys.setdefault(int(c), []).append(k)

        if fg_labels is None:
            fg_labels = sorted(class_to_keys.keys())
        else:
            fg_labels = [int(c) for c in fg_labels]

        if fg_class_sampling_weights is not None:
            assert fg_class_calibration_degree is None, \
                "Set at most one of fg_class_sampling_weights and fg_class_calibration_degree"
            assert len(fg_class_sampling_weights) == len(fg_labels), \
                f"fg_class_sampling_weights has length {len(fg_class_sampling_weights)} but there are " \
                f"{len(fg_labels)} foreground labels {fg_labels}"
            weights = [float(w) for w in fg_class_sampling_weights]
            self._sampling_mode = "manual"
        elif fg_class_calibration_degree is not None:
            # baseline f_c = case frequency; prob_c ~ f_c ** (1 - degree). Use raw case counts (renormalized later;
            # the dataset-size constant factor cancels). Absent classes get weight 0 and are dropped below, which
            # also avoids 0 ** negative when degree > 1.
            degree = float(fg_class_calibration_degree)
            weights = [(len(class_to_keys.get(c, [])) ** (1.0 - degree)) if len(class_to_keys.get(c, [])) > 0 else 0.0
                       for c in fg_labels]
            self._sampling_mode = f"calibrated(degree={degree})"
        else:
            weights = [1.0] * len(fg_labels)
            self._sampling_mode = "uniform"

        # keep only classes that are present in >=1 case and have a positive weight, then renormalize
        fg_classes, fg_weights = [], []
        for c, w in zip(fg_labels, weights):
            n_cases = len(class_to_keys.get(c, []))
            if n_cases == 0:
                print_to_log_file(f"[{self.__class__.__name__}] WARNING: foreground class {c} is not present in any case; "
                      f"dropping it from fg-class sampling.")
                continue
            if w <= 0:
                print_to_log_file(f"[{self.__class__.__name__}] foreground class {c} has weight {w} <= 0; "
                      f"excluding it from fg-class sampling.")
                continue
            fg_classes.append(c)
            fg_weights.append(w)

        total = float(sum(fg_weights))
        self._fg_classes = fg_classes
        self._fg_probs = [w / total for w in fg_weights] if total > 0 else []
        self.class_to_keys = {c: class_to_keys[c] for c in fg_classes}

        # natural_p is the baseline (degree 0) case-frequency ratio, shown alongside the actual sampling prob p
        total_cases = float(sum(len(self.class_to_keys[c]) for c in self._fg_classes))
        print_to_log_file(f"[{self.__class__.__name__}] fg-class sampling mode: {self._sampling_mode}; "
              f"over {len(self._fg_classes)} classes (case counts / natural_p -> p):")
        for c, p in zip(self._fg_classes, self._fg_probs):
            natural_p = len(self.class_to_keys[c]) / total_cases if total_cases > 0 else 0.0
            print_to_log_file(f"    class {c}: {len(self.class_to_keys[c])} cases, natural_p={natural_p:.4f} -> p={p:.4f}")
        if not self._fg_classes:
            print_to_log_file(f"[{self.__class__.__name__}] WARNING: no eligible foreground classes; "
                  "force-fg samples will fall back to nnUNet's default class selection.")

        # per-case "potential": within the chosen fg class, draw a case proportional to its weight. When no weight
        # groups are configured every case has weight 1, so we leave _class_to_key_probs None and fall back to a
        # uniform randint draw in _select_sample (cheaper and identical to the previous behavior).
        self._class_to_key_probs = None
        if case_sampling_weight_groups:
            self._class_to_key_probs = {}
            weight_counts = {}  # weight -> number of (distinct) cases, for the log
            for c in self._fg_classes:
                keys = self.class_to_keys[c]
                w = np.array([case_sampling_weight(k, case_sampling_weight_groups) for k in keys], dtype=np.float64)
                self._class_to_key_probs[c] = (w / w.sum()) if w.sum() > 0 else None
            for k in {k for c in self._fg_classes for k in self.class_to_keys[c]}:
                wk = case_sampling_weight(k, case_sampling_weight_groups)
                weight_counts[wk] = weight_counts.get(wk, 0) + 1
            print_to_log_file(f"[{self.__class__.__name__}] per-case sampling weights (foreground "
                  f"within-class draw); case counts by weight: "
                  f"{', '.join(f'{w}: {n}' for w, n in sorted(weight_counts.items()))}")

    def _select_sample(self, j, selected_keys):
        force_fg = self.get_do_oversample(j)
        if not force_fg or not self._fg_classes:
            # background sample, or no eligible fg classes -> keep the pre-sampled case and default class selection
            return selected_keys[j], force_fg, None
        # pick the foreground class first (weighted, default uniform), then a case that contains it, then let
        # get_bbox center the patch on a voxel of that class via overwrite_class
        c = self._fg_classes[np.random.choice(len(self._fg_classes), p=self._fg_probs)]
        keys = self.class_to_keys[c]
        key_probs = self._class_to_key_probs[c] if self._class_to_key_probs is not None else None
        if key_probs is not None:
            key = keys[np.random.choice(len(keys), p=key_probs)]
        else:
            key = keys[np.random.randint(len(keys))]
        return key, True, c
