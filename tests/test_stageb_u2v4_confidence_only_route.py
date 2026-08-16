import unittest

from models.GroundingDINO.groundingdino import (
    _validate_u2v2_confidence_only_route,
)


class U2V4ConfidenceOnlyRouteTest(unittest.TestCase):
    def test_u0_admission_route_is_valid_without_postgate_residual(self):
        _validate_u2v2_confidence_only_route(
            True, has_postgate_residual=False, has_u0_admission=True
        )

    def test_postgate_route_remains_valid(self):
        _validate_u2v2_confidence_only_route(
            True, has_postgate_residual=True, has_u0_admission=False
        )

    def test_missing_u2v2_route_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "requires a U2-v2 scoring route"):
            _validate_u2v2_confidence_only_route(
                True, has_postgate_residual=False, has_u0_admission=False
            )

    def test_flag_must_be_boolean(self):
        with self.assertRaises(TypeError):
            _validate_u2v2_confidence_only_route(
                1, has_postgate_residual=False, has_u0_admission=True
            )


if __name__ == "__main__":
    unittest.main()
