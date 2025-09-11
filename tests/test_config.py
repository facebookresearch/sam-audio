import unittest

from sam_audio.model.config import (
    SAM_AUDIO_CONFIGS,
    deserialize_config,
    serialize_config,
)


class TestConfig(unittest.TestCase):
    def test_serialization(self):
        config = SAM_AUDIO_CONFIGS["base-pe"]
        self.assertEqual(deserialize_config(serialize_config(config)), config)


if __name__ == "__main__":
    unittest.main()
