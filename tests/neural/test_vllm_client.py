from symcascade.neural.vllm_client import VLLMGemmaClient, _extract_top_logprob


class _Logprob:
    def __init__(self, logprob):
        self.logprob = logprob


class _Output:
    def __init__(self, text, logprobs):
        self.text = text
        self.logprobs = logprobs


class _RequestOutput:
    def __init__(self, text, logprobs):
        self.outputs = [_Output(text, logprobs)]


class FakeLLM:
    def __init__(self, text, top_logprob):
        self._text = text
        self._top = top_logprob
        self.received = []

    def generate(self, prompts, sp):
        self.received.append((prompts, sp))
        lp = [{0: _Logprob(self._top)}]  # one position, one token
        return [_RequestOutput(self._text, lp)]


def test_generate_returns_text_and_top_logprob():
    fake = FakeLLM(text="answer", top_logprob=-0.2)
    client = VLLMGemmaClient(llm=fake, sampling_params_factory=lambda: "SP")
    out = client.generate("prompt", enable_thinking=False)
    assert out["text"] == "answer"
    assert out["logprobs"] == [[-0.2]]
    assert fake.received[0][0] == ["prompt"]
    assert fake.received[0][1] == "SP"


def test_plugs_into_gemma_backend_confidence():
    from symcascade.neural.gemma_backend import GemmaBackend

    fake = FakeLLM(text="ok", top_logprob=0.0)  # exp(0)=1.0
    client = VLLMGemmaClient(llm=fake, sampling_params_factory=lambda: "SP")
    gemma = GemmaBackend(client=client, enable_thinking=False)
    resp = gemma.answer("q")
    assert resp.text == "ok"
    assert abs(resp.confidence - 1.0) < 1e-9


def test_extract_top_logprob_handles_empty_structures():
    assert _extract_top_logprob([]) == -10.0
    assert _extract_top_logprob([{}]) == -10.0
    assert _extract_top_logprob(None) == -10.0
    # picks the highest logprob among candidates at position 0
    lp = [{1: _Logprob(-3.0), 2: _Logprob(-0.5)}]
    assert _extract_top_logprob(lp) == -0.5


def test_construct_without_llm_does_not_import_vllm():
    import sys
    sys.modules.pop("vllm", None)
    VLLMGemmaClient()  # no llm, no factory -> nothing imported yet
    assert "vllm" not in sys.modules
