from collections import OrderedDict
import unittest

import torch

from nnunetv2.run.load_pretrained_weights import extract_pretrained_weights


class TestExtractPretrainedWeights(unittest.TestCase):
    def test_uses_primary_vesselfm_segmentation_network(self):
        primary_weight = torch.tensor([1.0])
        ema_weight = torch.tensor([2.0])
        checkpoint = {
            'state_dict': {
                'model.seg_net.encoder.weight': primary_weight,
                'ema_model.seg_net.encoder.weight': ema_weight,
            }
        }

        weights, source = extract_pretrained_weights(checkpoint)

        self.assertEqual(list(weights), ['encoder.weight'])
        self.assertIs(weights['encoder.weight'], primary_weight)
        self.assertIn("model.seg_net.", source)

    def test_falls_back_to_ema_vesselfm_segmentation_network(self):
        ema_weight = torch.tensor([2.0])
        checkpoint = {
            'state_dict': {
                'ema_model.seg_net.encoder.weight': ema_weight,
            }
        }

        weights, source = extract_pretrained_weights(checkpoint)

        self.assertEqual(list(weights), ['encoder.weight'])
        self.assertIs(weights['encoder.weight'], ema_weight)
        self.assertIn("ema_model.seg_net.", source)

    def test_keeps_native_nnunet_network_weights(self):
        native_weights = OrderedDict([('encoder.weight', torch.tensor([3.0]))])

        weights, source = extract_pretrained_weights({'network_weights': native_weights})

        self.assertIs(weights, native_weights)
        self.assertIn("network_weights", source)

    def test_rejects_unknown_lightning_state_dict(self):
        checkpoint = {'state_dict': {'model.other_network.weight': torch.tensor([1.0])}}

        with self.assertRaisesRegex(KeyError, 'model.seg_net'):
            extract_pretrained_weights(checkpoint)


if __name__ == '__main__':
    unittest.main()
