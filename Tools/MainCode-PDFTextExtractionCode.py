# import os
# import json
# import pdfplumber
# import openai
# from openai import OpenAI
# import jsonschema
# from pydantic import BaseModel, Field
# from crewai.tools.base_tool import BaseTool

# # Input schema for the PDF Extraction Tool
# class PDFTextExtractionInput(BaseModel):
#     pdf_path: str = Field(..., description="The file path of the PDF to extract text from.")

# class PDFTextExtractionTool(BaseTool):
#     name: str = "PDF Text Extraction Tool"
#     description: str = (
#         "Extracts text from a PDF file (ignoring tables), generates a dynamic JSON schema using an LLM "
#         "based on the extracted text, and converts the text into a JSON object conforming to that schema."
#     )
#     args_schema: type = PDFTextExtractionInput

#     def _run(self, pdf_path: str) -> dict:
#         """
#         Executes the full workflow:
#         1. Extracts text from the PDF while ignoring table regions.
#         2. Generates a dynamic JSON schema (or falls back to a baseline schema).
#         3. Converts the extracted text into a JSON object conforming to the chosen schema.
        
#         Returns:
#             A dictionary containing the extracted text, the schema used, and the converted JSON.
#         """
#         # Step 1: Extract text from PDF
#         extracted_text = self._extract_text_from_pdf(pdf_path)
        
#         # Step 2: Generate dynamic JSON schema using extracted text.
#         # Provide the path to your baseline JSON schema file.
#         baseline_schema_file = "path/to/baseline_schema.json"  # Update with your actual file path -> D:\AI Agents\Custom Tools\HDBFS\HDBFS_Text_json_schema.json
#         dynamic_schema = self._generate_dynamic_schema(extracted_text, baseline_schema_file)
        
#         # Step 3: Convert the extracted text to a JSON object conforming to the chosen schema.
#         converted_json = self._convert_text_to_json(extracted_text, dynamic_schema)
        
#         return {
#             "extracted_text": extracted_text,
#             "dynamic_schema": dynamic_schema,
#             "converted_json": converted_json
#         }

#     # ---------------------------
#     # Private Helper Methods
#     # ---------------------------
    
#     def _extract_text_from_pdf(self, pdf_path: str) -> str:
#         """
#         Extracts text from a PDF file while ignoring text within table boundaries.
#         """
#         all_text = []
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 page_tables = page.extract_tables()
#                 table_bboxes = [table.bbox for table in page.find_tables()] if page_tables else []
                
#                 def not_within_bboxes(obj):
#                     def obj_in_bbox(bbox):
#                         v_mid = (obj["top"] + obj["bottom"]) / 2
#                         h_mid = (obj["x0"] + obj["x1"]) / 2
#                         x0, top, x1, bottom = bbox
#                         return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
#                     return not any(obj_in_bbox(bbox) for bbox in table_bboxes)
                
#                 text = page.filter(not_within_bboxes).extract_text()
#                 if text:
#                     all_text.append(text)
#         full_text = "\n".join(all_text)
#         return full_text if full_text else "No text found outside tables."

#     def _load_baseline_schema(self, file_path: str) -> dict:
#         """
#         Loads the baseline JSON schema from a file.
#         """
#         with open(file_path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     def _generate_prompt(self, extracted_text: str, baseline: dict) -> str:
#         """
#         Constructs a prompt for the LLM that includes the extracted text and the baseline schema.
#         """
#         prompt = (
#             "Using the following extracted text from a PDF document:\n\n"
#             f"{extracted_text}\n\n"
#             "and the baseline JSON schema provided below:\n\n"
#             f"{json.dumps(baseline, indent=2)}\n\n"
#             "Generate a JSON schema that best fits the content of extracted text from the document. "
#             "If the extracted text is consistent with the baseline schema, adjust accordingly. "
#             "Otherwise, use the baseline schema. Provide only the JSON schema in your response."
#         )
#         return prompt

#     def _call_llm(self, prompt: str) -> str:
#         """
#         Calls the LLM using OpenAI's API to generate a JSON schema or to convert text based on the provided prompt.
#         """
#         try:
#             client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",  # Specify your chosen model
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=1000,
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             print("Error during LLM call:", e)
#             return ""

#     def _is_valid_json_schema(self, schema: dict) -> bool:
#         """
#         Validates the given JSON schema against JSON Schema Draft 7.
#         """
#         try:
#             jsonschema.Draft7Validator.check_schema(schema)
#             return True
#         except jsonschema.exceptions.SchemaError as error:
#             print("Generated schema is invalid:", error)
#             return False

#     def _generate_dynamic_schema(self, extracted_text: str, baseline_file: str) -> dict:
#         """
#         Generates a dynamic JSON schema by:
#         1. Loading the baseline schema from a file.
#         2. Building a prompt that includes the extracted text and baseline schema.
#         3. Calling the LLM to produce a new schema.
#         4. Validating the new schema and falling back to the baseline if necessary.
#         """
#         baseline = self._load_baseline_schema(baseline_file)
#         prompt = self._generate_prompt(extracted_text, baseline)
#         generated_schema_str = self._call_llm(prompt)
#         try:
#             generated_schema = json.loads(generated_schema_str)
#         except json.JSONDecodeError as error:
#             print("Error decoding generated schema:", error)
#             return baseline

#         if self._is_valid_json_schema(generated_schema):
#             return generated_schema
#         else:
#             return baseline

#     def _convert_text_to_json(self, extracted_text: str, schema: dict) -> dict:
#         """
#         Converts the extracted text into a JSON object that conforms to the provided JSON schema.
        
#         This method builds a prompt that instructs the LLM to transform the extracted text into a JSON object 
#         following the schema. It then calls the LLM using the internal _call_llm helper and attempts to parse 
#         the response as JSON.
        
#         Parameters:
#             extracted_text (str): The text extracted from the PDF.
#             schema (dict): The JSON schema (either dynamic or baseline) to be followed.
        
#         Returns:
#             dict: A JSON object adhering to the schema. Returns an empty dict if parsing fails.
#         """
#         prompt = (
#             "Convert the following extracted text into a JSON object that strictly conforms to the provided JSON schema. "
#             "Ensure that the output adheres exactly to the schema and contains only the JSON object.\n\n"
#             "Extracted Text:\n"
#             f"{extracted_text}\n\n"
#             "JSON Schema:\n"
#             f"{json.dumps(schema, indent=2)}\n\n"
#         )
        
#         json_response = self._call_llm(prompt)
        
#         try:
#             converted_json = json.loads(json_response)
#             return converted_json
#         except json.JSONDecodeError as e:
#             print("Failed to parse JSON from LLM response:", e)
#             return {}
    

































import os
import json
import pdfplumber
import openai
from openai import OpenAI
import jsonschema
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool

# Input schema for the PDF Extraction Tool
class PDFTextExtractionInput(BaseModel):
    pdf_path: str = Field(..., description="The file path of the PDF to extract text from.")
    baseline_schema_path: str = Field(..., description="The file path of the predefined baseline JSON schema.")


class PDFTextExtractionTool(BaseTool):
    name: str = "PDF Text Extraction Tool"
    description: str = (
        "Extracts text from a PDF file (ignoring tables), loads a predefined JSON schema, "
        "prompts the LLM to map the extracted text to that schema, and validates the resulting JSON."
    )
    args_schema: type = PDFTextExtractionInput

    def _run(self, pdf_path: str, baseline_schema_path: str) -> dict:
        """
        Executes the full workflow:
          1. Extracts text from the PDF while ignoring table regions.
          2. Loads the predefined JSON schema.
          3. Prompts the LLM to map the extracted text to the JSON schema.
          4. Validates the output JSON against the schema.
        
        Returns:
            A dictionary containing the extracted text, the schema used, the converted JSON, and its validation status.
        """
        # Step 1: Extract text from PDF
        extracted_text = self._extract_text_from_pdf(pdf_path)
        
        # Step 2: Load the predefined JSON schema (update the file path accordingly)
        baseline_schema = self._load_baseline_schema(baseline_schema_path)
        
        # Step 3: Convert the extracted text to a JSON object using the predefined schema.
        converted_json = self._convert_text_to_json(extracted_text, baseline_schema)
        
        # Step 4: Validate the output JSON against the baseline schema.
        valid = False
        try:
            jsonschema.validate(converted_json, baseline_schema)
            valid = True
        except jsonschema.ValidationError as e:
            print("Validation error:", e)
        
        return {
            "extracted_text": extracted_text,
            "schema_used": baseline_schema,
            "converted_json": converted_json,
            "valid": valid
        }
    
    # ---------------------------
    # Private Helper Methods
    # ---------------------------
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file while ignoring text within table boundaries.
        """
        all_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Get bounding boxes of detected tables on the page
                page_tables = page.extract_tables()
                table_bboxes = [table.bbox for table in page.find_tables()] if page_tables else []
                
                # Function to filter out text objects within table boundaries
                def not_within_bboxes(obj):
                    def obj_in_bbox(bbox):
                        v_mid = (obj["top"] + obj["bottom"]) / 2
                        h_mid = (obj["x0"] + obj["x1"]) / 2
                        x0, top, x1, bottom = bbox
                        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
                    return not any(obj_in_bbox(bbox) for bbox in table_bboxes)
                
                text = page.filter(not_within_bboxes).extract_text()
                if text:
                    all_text.append(text)
        full_text = "\n".join(all_text)
        return full_text if full_text else "No text found outside tables."

    def _load_baseline_schema(self, file_path: str) -> dict:
        """
        Loads the predefined JSON schema from a file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM using OpenAI's API to convert the extracted text into a JSON object.
        """
        try:
            # Initialize the OpenAI client with your API key from the environment variables.
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Specify your chosen model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error during LLM call:", e)
            return ""
    
    def _convert_text_to_json(self, extracted_text: str, schema: dict) -> dict:
        """
        Converts the extracted text into a JSON object that conforms to the provided JSON schema.
        
        Constructs a prompt that instructs the LLM to transform the extracted text into a JSON object 
        following the schema. Calls the LLM using the _call_llm helper and parses the response.
        
        Parameters:
            extracted_text (str): The text extracted from the PDF.
            schema (dict): The JSON schema that the output must follow.
        
        Returns:
            dict: A JSON object adhering to the schema. Returns an empty dict if parsing fails.
        """
        prompt = (
            "You are a JSON Response Generator. Your task is to convert the plain text extracted from the PDF into a JSON object by using the provided JSON schema. "
            "Ensure that the output adheres exactly to the provided JSON schema and contains only the JSON object mapped to the extracted text.\n\n"
            "Extracted Text:\n"
            f"{extracted_text}\n\n"
            "JSON Schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
        )
        
        json_response = self._call_llm(prompt)
        
        try:
            converted_json = json.loads(json_response)
            return converted_json
        except json.JSONDecodeError as e:
            print("Failed to parse JSON from LLM response:", e)
            return {}
