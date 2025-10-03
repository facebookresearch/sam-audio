import unittest

from sam_audio.model.config import (
    SAMAudioConfig,
    TransformerConfig,
    deserialize_config,
    serialize_config,
)


class TestConfig(unittest.TestCase):
    def test_serialization(self):
        config = SAMAudioConfig(transformer=TransformerConfig(n_layers=100))
        self.assertEqual(deserialize_config(serialize_config(config)), config)


if __name__ == "__main__":
    unittest.main()
