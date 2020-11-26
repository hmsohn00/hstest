import os

__version__ = "0.2"

package_path = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))
package_name = os.path.basename(package_path)
schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "settings-schema.json")