"""ALFWorld PDDL domain (embedded).

ALFWorld ships a fixed PDDL domain for its 6 task classes (pick, clean,
heat, cool, examine, pick-two). We embed a faithful, simplified domain
string so experiments don't depend on the alfworld package being importable
at module load — the adapter loads tasks lazily and falls back to this
domain when the alfworld resource isn't on disk.

Predicates: at-agent, at-object, in, holding, clean, hot, cold, examined,
location, object, agent, is-hot-source, is-cold-source, is-examined-source.
Actions: goto, pick, put, clean, heat, cool, examine.
"""
from __future__ import annotations

# ALFWorld task classes — used as labels for skeleton grouping and ablation.
TASK_CLASSES = (
    "pick_and_place",
    "pick_clean_then_place",
    "pick_heat_then_place",
    "pick_cool_then_place",
    "look_at_obj_in_light",
    "pick_two_and_place",
)

# Canonical action sequence per task class. These are the skeletons C2 caches
# and L1 reuses via constrained replanning; the generalized placeholder form
# matches Skeleton generalization (OBJ/LOC tokens).
TASK_SKELETONS: dict[str, tuple[str, ...]] = {
    "pick_and_place": ("goto", "pick", "goto", "put"),
    "pick_clean_then_place": ("goto", "pick", "goto", "clean", "goto", "put"),
    "pick_heat_then_place": ("goto", "pick", "goto", "heat", "goto", "put"),
    "pick_cool_then_place": ("goto", "pick", "goto", "cool", "goto", "put"),
    "look_at_obj_in_light": ("goto", "pick", "goto", "examine"),
    "pick_two_and_place": ("goto", "pick", "goto", "put",
                           "goto", "pick", "goto", "put"),
}

# Embedded domain. Faithful to ALFWorld's symbolic backend; object/room
# constants are per-problem, so only predicates + action signatures live here.
DOMAIN_PDDL = """\
(define (domain alfworld)
  (:requirements :strips :typing :negative-preconditions)
  (:types agent location object)
  (:predicates
    (at-agent ?a - agent ?l - location)
    (at-object ?o - object ?l - location)
    (in ?o - object ?l - location)
    (holding ?a - agent ?o - object)
    (clean ?o - object)
    (hot ?o - object)
    (cold ?o - object)
    (examined ?o - object)
    (is-hot-source ?l - location)
    (is-cold-source ?l - location)
    (is-examined-source ?l - location)
  )
  (:action goto
    :parameters (?a - agent ?l - location)
    :precondition (agent ?a)
    :effect (forall (?o - object) (when (holding ?a ?o) (at-object ?o ?l)))
  )
  (:action pick
    :parameters (?a - agent ?o - object ?l - location)
    :precondition (and (at-agent ?a ?l) (at-object ?o ?l))
    :effect (and (holding ?a ?o) (not (at-object ?o ?l)))
  )
  (:action put
    :parameters (?a - agent ?o - object ?l - location)
    :precondition (and (at-agent ?a ?l) (holding ?a ?o))
    :effect (and (at-object ?o ?l) (not (holding ?a ?o)))
  )
  (:action clean
    :parameters (?a - agent ?o - object ?l - location)
    :precondition (and (at-agent ?a ?l) (holding ?a ?o))
    :effect (clean ?o)
  )
  (:action heat
    :parameters (?a - agent ?o - object ?l - location)
    :precondition (and (at-agent ?a ?l) (holding ?a ?o) (is-hot-source ?l))
    :effect (hot ?o)
  )
  (:action cool
    :parameters (?a - agent ?o - object ?l - location)
    :precondition (and (at-agent ?a ?l) (holding ?a ?o) (is-cold-source ?l))
    :effect (cold ?o)
  )
  (:action examine
    :parameters (?a - agent ?o - object ?l - location)
    :precondition (and (at-agent ?a ?l) (holding ?a ?o) (is-examined-source ?l))
    :effect (examined ?o)
  )
)
"""
