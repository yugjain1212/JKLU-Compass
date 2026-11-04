"""Test the local Ollama LLM client."""

from jklu_compass.generation.local_llm import LocalLLM, LocalLLMError


def main() -> None:
    llm = LocalLLM()

    prompt = """
You are JKLU Compass, an assistant that answers questions about
JK Lakshmipat University.

Answer the following question in one short sentence:
What information should a university assistant provide to students?

Do not invent specific university facts.
"""

    try:
        answer = llm.generate(prompt)
        print("\n--- Local LLM Response ---\n")
        print(answer)

    except LocalLLMError as exc:
        print(f"\nLLM Error: {exc}")


if __name__ == "__main__":
    main()
