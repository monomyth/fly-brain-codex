"""Versioned camera registration. Old and new viewpoints must never mix silently."""
LEGACY_CAMERAS = ("Front", "Top")
CURRENT_CAMERAS = ("Front", "Gripper")
CURRENT_RIG = "front336l-gripper305-rgb-v4"
LEGACY_RIG = "legacy-front-top-v1"


def image_contract(images):
    names = tuple(image["name"] for image in images)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Exactly two distinct synchronized observation cameras are required")
    revisions = {image.get("rig_revision", LEGACY_RIG) for image in images}
    if len(revisions) != 1:
        raise ValueError("Camera rig revisions were mixed")
    revision = next(iter(revisions))
    pair = CURRENT_CAMERAS if revision == CURRENT_RIG else LEGACY_CAMERAS
    if revision not in (CURRENT_RIG, LEGACY_RIG) or set(names) != set(pair):
        raise ValueError("Camera pair does not match its rig revision")
    return pair, revision


def observation_contract(observation):
    pair, revision = image_contract(observation["images"])
    if observation.get("camera_rig_revision", revision) != revision:
        raise ValueError("Observation and image camera revisions disagree")
    if tuple(observation.get("observation_camera_names", pair)) != pair:
        raise ValueError("Observation and image camera names disagree")
    return pair, revision
