import brevettiai.interfaces.vue_schema_utils as vue
from .hstest import Hstest

if __name__ == "__main__":
    import argparse
    from . import constants

    parser = argparse.ArgumentParser()
    parser.add_argument('--schema_path', help='Output location of schema', default=constants.schema_path)
    args = parser.parse_args()

    print(f"Generating application schema to '{args.schema_path}'")
    vue.generate_application_schema(Hstest.Settings.get_schema(), path=args.schema_path)

