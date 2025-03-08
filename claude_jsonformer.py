from pydantic import BaseModel, Field
import json
from typing import Any
import anthropic


class ClaudeJsonformer:
    def __init__(
        self,
        *,
        pydantic_model: type[BaseModel] = None,
        json_schema: dict[str, Any] = None,
        debug: bool = False,
    ):
        """Only 1 of pydantic_model or json_schema must be provided."""

        if json_schema is not None and pydantic_model is not None:
            raise ValueError(
                "Only 1 of pydantic_model or json_schema must be provided."
            )

        self.client = anthropic.Client()
        self.debug_on = debug
        self.pydantic_model = pydantic_model
        self.json_schema = (
            json_schema
            if json_schema is not None
            else pydantic_model.model_json_schema()
        )

    def generate(self, prompt: str):
        """Generate JSON data based on the schema using Claude.

        Returns a Pydantic model if pydantic_model was provided, otherwise returns a dict."""

        system_prompt = f"""You are a helpful assistant that generates JSON data based on provided schemas.
        You must ONLY output valid JSON that matches the schema exactly.
        Do not include any explanation or additional text. Generate JSON data matching exactly this schema:
        {json.dumps(self.json_schema, indent=2)}
        
        Output only the JSON data, nothing else."""

        user_prompt = prompt

        response = self.client.messages.create(
            model="claude-3-7-sonnet-latest",
            temperature=0.0,  # No randomness
            system=system_prompt,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract JSON from response
        try:
            response_text = response.content[0].text
            result = json.loads(response_text)

            if self.pydantic_model is not None:
                # enforce model validation
                return self.pydantic_model.model_validate(result)
            else:
                return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from Claude response: {e}")


# Example usage:
if __name__ == "__main__":

    class User(BaseModel):
        name: str
        age: int
        is_active: bool
        hobbies: list[str] = Field(default_factory=list)

    jsonformer = ClaudeJsonformer(
        pydantic_model=User,
    )
    result = jsonformer.generate(
        "Generate data for a young software developer",
    )
    print(result.model_dump_json(indent=2))
