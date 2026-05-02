"""Gantry setup step helpers."""


def open_gantry(use_mock=False, start_x=0.0, start_y=0.0):
    if use_mock:
        from hardware.mock_gantry import MockGantry

        return MockGantry(start_x=start_x, start_y=start_y)

    from config import GRBL_PORT
    from hardware.gantry import Gantry

    return Gantry(GRBL_PORT)


def home_gantry(gantry):
    gantry.home()
    return gantry.get_position()


def move_to_survey(gantry):
    from config import SURVEY_POS_X, SURVEY_POS_Y

    gantry.move_absolute(SURVEY_POS_X, SURVEY_POS_Y)
    return gantry.get_position()
