import unittest

from hstest.hstest import Hstest

class TestHstest(unittest.TestCase):

    minimal_settings = {
        "image": {
            "input_width": 128,
            "input_height": 128,
            "color_mode": "rgb"
        },
    }

    def test_generate_application_schema(self):
        schema = Hstest.Settings.get_schema().schema
        assert isinstance(schema, dict), 'schema is not dict'
        assert "fields" in schema
        assert isinstance(schema["fields"], (list, tuple))

    def test_initialize_application(self):
        job = Hstest(
            schema=Hstest.Settings.get_schema(),
            name="HSTest model",
            id="id",
            job_dir="",
            settings=self.minimal_settings,
            tags=[],
            datasets=[],
            api_key="Secret key",
            charts_url="charts_url",
            complete_url="complete_url",
            remote_url="remote_url",
            host_name="host_name",
        )
        assert job.settings.epochs > 0
        assert job.settings.batch_size > 0


if __name__ == '__main__':
    unittest.main()
