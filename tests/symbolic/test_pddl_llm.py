from symcascade.symbolic.pddl_llm import LLMPDDLGenerator, extract_pddl


class FakeLLM:
    def __init__(self, output):
        self._output = output
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self._output


DOMAIN = "(define (domain d) ...)"


def test_extract_pddl_from_fenced_block():
    text = "Here:\n```pddl\n(define (problem p))\n```\nDone."
    assert extract_pddl(text) == "(define (problem p))"


def test_extract_pddl_falls_back_to_raw_when_no_fence():
    assert extract_pddl("(define (problem p))") == "(define (problem p))"


def test_generate_pddl_extracts_and_returns_problem():
    fake = FakeLLM(output="```pddl\n(define (problem p1))\n```")
    gen = LLMPDDLGenerator(llm=fake, domain_pddl=DOMAIN)
    assert gen.generate_pddl("put apple in fridge") == "(define (problem p1))"
    assert "Domain:" in fake.calls[0]
    assert "put apple in fridge" in fake.calls[0]


def test_generate_pddl_returns_empty_when_validator_rejects():
    fake = FakeLLM(output="(define (problem p1))")
    gen = LLMPDDLGenerator(
        llm=fake, domain_pddl=DOMAIN,
        validator=lambda dom, prob: False,
    )
    assert gen.generate_pddl("goal") == ""


def test_generate_pddl_passes_when_validator_accepts():
    fake = FakeLLM(output="(define (problem p1))")
    seen = []
    gen = LLMPDDLGenerator(
        llm=fake, domain_pddl=DOMAIN,
        validator=lambda dom, prob: (seen.append((dom, prob)) or True),
    )
    assert gen.generate_pddl("goal") == "(define (problem p1))"
    assert seen == [(DOMAIN, "(define (problem p1))")]


def test_plugs_into_pddl_gen_stage():
    from symcascade.core.types import Query
    from symcascade.symbolic.pddl_gen import PDDLGenStage
    from symcascade.cache.skeleton_cache import SkeletonCache

    fake = FakeLLM(output="```pddl\n(define (problem p1))\n```")
    gen = LLMPDDLGenerator(llm=fake, domain_pddl=DOMAIN)

    class FakeFD:
        def solve(self, domain, problem):
            class R:
                success = True
                plan = [{"name": "act"}]
            return R()

    stage = PDDLGenStage(
        llm=gen, fd=FakeFD(), cache=SkeletonCache(),
        domain_pddl=DOMAIN, problem_pddl_fn=lambda q: "",
    )
    r = stage.run(Query(text="goal"))
    assert r.success is True
