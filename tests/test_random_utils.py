import importlib.util
from pathlib import Path
import unittest


# Import this dependency-free module directly so the focused test does not
# require the optional model stack imported by rewardbench.__init__.
module_path = Path(__file__).parents[1] / "rewardbench" / "random_utils.py"
module_spec = importlib.util.spec_from_file_location("rewardbench_random_utils", module_path)
assert module_spec is not None and module_spec.loader is not None
random_utils = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(random_utils)
generate_shuffle_positions = random_utils.generate_shuffle_positions


class RandomUtilsTest(unittest.TestCase):
    def test_positions_are_reproducible_for_a_seed(self):
        self.assertEqual(generate_shuffle_positions(32, 17), generate_shuffle_positions(32, 17))

    def test_positions_are_in_the_four_valid_slots(self):
        positions = generate_shuffle_positions(128, 17)

        self.assertEqual(len(positions), 128)
        self.assertTrue(all(position in range(4) for position in positions))

    def test_negative_example_count_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_shuffle_positions(-1, 17)


if __name__ == "__main__":
    unittest.main()
