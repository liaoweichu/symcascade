from symcascade.symbolic.skeleton import Skeleton, SkeletonAction, extract_skeleton


def test_extract_skeleton_from_action_sequence():
    plan = [
        {"name": "goto", "pre": ["at-agent room1"], "eff": ["at-agent room2"]},
        {"name": "take", "pre": ["at-object apple room2"], "eff": ["holding apple"]},
        {"name": "goto", "pre": ["at-agent room2"], "eff": ["at-agent room3"]},
        {"name": "put", "pre": ["holding apple", "at-agent room3"], "eff": ["at-object apple room3"]},
    ]
    skel = extract_skeleton(plan)
    assert [a.name for a in skel.actions] == ["goto", "take", "goto", "put"]
    # predicates are generalized: object/room identifiers stripped
    assert skel.actions[0].pre == ("at-agent LOC",)
    assert skel.actions[1].pre == ("at-object OBJ LOC",)
    assert skel.actions[3].eff == ("at-object OBJ LOC",)


def test_skeleton_template_round_trips():
    skel = Skeleton(actions=[
        SkeletonAction(name="take", pre=["at-object OBJ LOC"], eff=["holding OBJ"]),
    ])
    s = skel.to_template_str()
    assert "take" in s
    parsed = Skeleton.from_template_str(s)
    assert parsed.actions[0].name == "take"


def test_two_plans_same_shape_same_skeleton():
    plan_a = [{"name": "goto", "pre": ["at-agent r1"], "eff": ["at-agent r2"]},
              {"name": "take", "pre": ["at-object x r2"], "eff": ["holding x"]}]
    plan_b = [{"name": "goto", "pre": ["at-agent r3"], "eff": ["at-agent r4"]},
              {"name": "take", "pre": ["at-object y r4"], "eff": ["holding y"]}]
    assert extract_skeleton(plan_a).to_template_str() == extract_skeleton(plan_b).to_template_str()
