# import os
# import io
# import json
# import base64
# import pdfplumber
# import jsonschema
# from PIL import Image
# from typing import Any, List, Dict, Optional, Type
# from pydantic import BaseModel, Field
# from crewai.tools import BaseTool  # Adjust this import according to your project structure
# from openai import OpenAI  # Ensure the OpenAI client is installed and imported

# # Input schema for the custom tool
# class PDFImageTextExtractionToolSchema(BaseModel):
#     pdf_path: str = Field(..., description="Path to the PDF file to process")
#     output_folder: str = Field(..., description="Folder to save extracted images")
#     max_image_size: int = Field(1024, description="Maximum size for resizing images")
#     baseline_schema_path: Optional[str] = Field(
#         None, description="Path to the baseline JSON schema file"
#     )
#     # Additional fields can be added if needed

# class PDFImageTextExtractionTool(BaseTool):
#     name: str = "PDF Image Text Extraction Tool"
#     description: str = (
#         "Extracts images from a PDF, resizes and encodes them to base64, extracts text "
#         "from the images using GPT-4o-mini, generates a dynamic JSON schema, converts the "
#         "extracted text to JSON, and validates the JSON."
#     )
#     args_schema: Type[BaseModel] = PDFImageTextExtractionToolSchema

#     def _run(
#         self,
#         pdf_path: str,
#         output_folder: str,
#         max_image_size: int = 1024,
#         baseline_schema_path: Optional[str] = None,
#     ) -> Any:
#         """
#         Orchestrates the full PDF processing workflow.
#         If a baseline schema path is provided, the dynamic schema is generated and used to convert
#         the extracted text into JSON. Otherwise, the tool returns the image-based text JSON.
#         """
#         # Step 1: Extract images and obtain a JSON string of texts extracted via GPT-4o-mini
#         image_text_json_str = self._get_image_text_json(pdf_path, output_folder, max_image_size)
        
#         # Step 2: Additionally, extract texts from images as a list (for dynamic schema generation)
#         images = self._extract_images_from_pdf(pdf_path, output_folder, max_image_size)
#         extracted_texts = self._extract_text_from_images(images)
        
#         if baseline_schema_path:
#             # Generate a dynamic schema using the extracted texts and a baseline schema file
#             dynamic_schema = self._generate_dynamic_schema(extracted_texts, baseline_schema_path)
#             print("Dynamic schema generated:", dynamic_schema)
#             # Convert the combined extracted text into JSON using the dynamic schema
#             json_data = self._convert_text_to_json("\n".join(extracted_texts), dynamic_schema)
#             self._validate_json(json_data, dynamic_schema)
#             return json_data

#         # Fallback: return the image text JSON as-is.
#         return image_text_json_str

#     def _extract_images_from_pdf(
#         self, pdf_path: str, output_folder: str, max_image_size: int = 1024
#     ) -> List[Dict[str, str]]:
#         """
#         Extracts images from a PDF using pdfplumber, saves them as PNG files in the output folder,
#         and returns a list of dictionaries with file paths and base64 encoded images.
#         """
#         try:
#             def clip_bbox(bbox, parent_bbox):
#                 x0, top, x1, bottom = bbox
#                 p_x0, p_top, p_x1, p_bottom = parent_bbox
#                 new_x0 = max(x0, p_x0)
#                 new_top = max(top, p_top)
#                 new_x1 = min(x1, p_x1)
#                 new_bottom = min(bottom, p_bottom)
#                 return (new_x0, new_top, new_x1, new_bottom)
            
#             os.makedirs(output_folder, exist_ok=True)
#             image_counter = 1
#             extracted_images = []
#             with pdfplumber.open(pdf_path) as pdf:
#                 for page in pdf.pages:
#                     parent_bbox = page.bbox
#                     for image in page.images:
#                         bbox = (image["x0"], image["top"], image["x1"], image["bottom"])
#                         clipped_bbox = clip_bbox(bbox, parent_bbox)
#                         try:
#                             cropped_page = page.crop(clipped_bbox)
#                             page_image = cropped_page.to_image(resolution=300, antialias=True)
#                         except Exception as e:
#                             print(f"Error extracting image on page {page.page_number}: {e}")
#                             continue
#                         object_id = image.get("object_id", image_counter)
#                         image_filename = f"page_{page.page_number}_img_{object_id}.png"
#                         file_path = os.path.join(output_folder, image_filename)
#                         try:
#                             page_image.save(file_path, format="PNG", quantize=False)
#                             print(f"Saved high quality image to {file_path}")
#                             base64_encoded = self._resize_and_encode_image(file_path, max_image_size)
#                             extracted_images.append({"file_path": file_path, "base64": base64_encoded})
#                         except Exception as e:
#                             print(f"Error saving image on page {page.page_number}: {e}")
#                         image_counter += 1
#             return extracted_images
#         except Exception as e:
#             print(f"Error in _extract_images_from_pdf: {e}")
#             return []

#     def _resize_and_encode_image(
#         self, image_path: str, max_image_size: int = 1024
#     ) -> Optional[str]:
#         """
#         Loads an image, resizes it while maintaining aspect ratio, and converts it to base64 encoding.
#         """
#         try:
#             with Image.open(image_path) as img:
#                 width, height = img.size
#                 if width > height:
#                     max_size = (max_image_size, int(max_image_size * (height / width)))
#                 else:
#                     max_size = (int(max_image_size * (width / height)), max_image_size)
#                 img = img.resize(max_size, Image.LANCZOS)
#                 buffered = io.BytesIO()
#                 img.save(buffered, format="JPEG")
#                 img_bytes = buffered.getvalue()
#                 base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
#                 return base64_encoded_image
#         except Exception as e:
#             print(f"Error in _resize_and_encode_image: {e}")
#             return None

#     def _get_image_text_json(
#         self, pdf_path: str, output_folder: str, max_image_size: int = 1024
#     ) -> str:
#         """
#         Extracts images from a PDF, extracts text from each image using GPT-4o-mini,
#         and returns a JSON-formatted string conforming to a predefined schema.
#         """
#         try:
#             images_info = self._extract_images_from_pdf(pdf_path, output_folder, max_image_size)
#             if not images_info:
#                 return json.dumps({"extracted_texts": []})
            
#             client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#             extracted_texts = []
#             for img_info in images_info:
#                 file_path = img_info.get("file_path")
#                 base64_image = img_info.get("base64")
#                 image_name = os.path.basename(file_path)
#                 if not base64_image:
#                     extracted_texts.append({
#                         "image_name": image_name,
#                         "content": "Error: No base64 image data available."
#                     })
#                     continue
#                 image_data = f"data:image/jpeg;base64,{base64_image}"
#                 try:
#                     response = client.chat.completions.create(
#                         model="gpt-4o-mini",
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     {"type": "text", "text": "Extract each and every textual details from the image."},
#                                     {"type": "image_url", "image_url": {"url": image_data}},
#                                 ],
#                             }
#                         ],
#                         max_tokens=300,
#                     )
#                     text = response.choices[0].message.content
#                 except Exception as e:
#                     text = f"Error processing image: {str(e)}"
#                 extracted_texts.append({
#                     "image_name": image_name,
#                     "content": text
#                 })
#             output_json = {"extracted_texts": extracted_texts}
#             return json.dumps(output_json)
#         except Exception as e:
#             print(f"Error in _get_image_text_json: {e}")
#             return json.dumps({"extracted_texts": []})

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
#         Calls the LLM using OpenAI's API to generate a JSON schema or convert text based on the provided prompt.
#         """
#         try:
#             client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=4000,
#                 top_p=0.5
#             )
#             resp = response.choices[0].message.content
#             print(resp)
#             return resp
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

    # def _generate_dynamic_schema(self, extracted_texts: List[str], baseline_file: str) -> dict:
    #     """
    #     Generates a dynamic JSON schema by:
    #       1. Loading the baseline schema from a file.
    #       2. Building a prompt that includes the extracted text and baseline schema.
    #       3. Calling the LLM to produce a new schema.
    #       4. Validating the new schema and falling back to the baseline if necessary.
    #     """
    #     combined_text = "\n".join(extracted_texts)
    #     baseline = self._load_baseline_schema(baseline_file)
    #     prompt = self._generate_prompt(combined_text, baseline)
    #     generated_schema_str = self._call_llm(prompt)
    #     try:
    #         generated_schema = json.loads(generated_schema_str)
    #     except json.JSONDecodeError as error:
    #         print("Error decoding generated schema:", error)
    #         return baseline

    #     if self._is_valid_json_schema(generated_schema):
    #         return generated_schema
    #     else:
    #         return baseline

    # def _extract_text_from_images(self, images: List[Dict[str, str]]) -> List[str]:
    #     """
    #     Placeholder method to extract text from images using OCR.
    #     """
    #     extracted_texts = []
    #     for img in images:
    #         extracted_texts.append(f"Extracted text from {img.get('file_path', '')}")
    #     return extracted_texts

    # def _convert_text_to_json(self, extracted_text: str, schema: dict) -> dict:
    #     """
    #     Converts the extracted text into a JSON object that conforms to the provided JSON schema.
        
    #     This method builds a prompt that instructs the LLM to transform the extracted text into a JSON object 
    #     following the schema. It then calls the LLM using the internal _call_llm helper and attempts to parse 
    #     the response as JSON.
        
    #     Parameters:
    #         extracted_text (str): The text extracted from the PDF.
    #         schema (dict): The JSON schema (either dynamic or baseline) to be followed.
        
    #     Returns:
    #         dict: A JSON object adhering to the schema. Returns an empty dict if parsing fails.
    #     """
    #     prompt = (
    #         "Convert the following extracted text into a JSON object that strictly conforms to the provided JSON schema. "
    #         "Ensure that the output adheres exactly to the schema and contains only the JSON object.\n\n"
    #         "Extracted Text:\n"
    #         f"{extracted_text}\n\n"
    #         "JSON Schema:\n"
    #         f"{json.dumps(schema, indent=2)}\n\n"
    #     )
        
    #     json_response = self._call_llm(prompt)
        
    #     try:
    #         converted_json = json.loads(json_response)
    #         return converted_json
    #     except json.JSONDecodeError as e:
    #         print("Failed to parse JSON from LLM response:", e)
    #         return {}

    # def _validate_json(self, json_data: dict, schema: dict) -> None:
    #     """
    #     Placeholder method to validate that the JSON data complies with the provided schema.
    #     """
    #     print("JSON validated against schema.")


































import os
import io
import json
import base64
import pdfplumber
import jsonschema
from PIL import Image
from typing import List, Dict, Optional, Type, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool  # Adjust this import according to your project structure
from openai import OpenAI  # Ensure the OpenAI client is installed and imported

# Input schema for the custom tool
class PDFImageTextExtractionToolSchema(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file to process")
    output_folder: str = Field(..., description="Folder to save extracted images")
    max_image_size: int = Field(1024, description="Maximum size for resizing images")
    baseline_schema_path: Optional[str] = Field(
        None, description="Path to the baseline JSON schema file"
    )

class PDFImageTextExtractionTool(BaseTool):
    name: str = "PDF Image Text Extraction Tool"
    description: str = (
        "Extracts images from a PDF, resizes and encodes them to base64, extracts text "
        "from the images using an LLM, and maps the extracted text to a JSON format "
        "according to a predefined schema via an LLM."
    )
    args_schema: Type[BaseModel] = PDFImageTextExtractionToolSchema

    def _run(
        self,
        pdf_path: str,
        output_folder: str,
        max_image_size: int = 1024,
        baseline_schema_path: Optional[str] = None,
    ) -> Any:
        """
        Orchestrates the workflow:
          1. Extract images from the PDF.
          2. Resize and encode images to base64.
          3. Extract text from the images using the LLM.
          4. Map the extracted text to JSON using a predefined schema (via the LLM).
        """
        # Step 1: Extract images (with base64 encoding)
        images_info = self._extract_images_from_pdf(pdf_path, output_folder, max_image_size)
        # Step 2: Extract text from the images using the LLM
        image_text_json = self._get_image_text_json(images_info)
        # Step 3: Load the predefined (baseline) schema using _load_baseline_schema if a path is provided
        if baseline_schema_path:
            predefined_schema = self._load_baseline_schema(baseline_schema_path)
        else:
            # Fallback predefined schema inline if no file is provided
            predefined_schema = {
                "type": "object",
                "properties": {
                    "extracted_texts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "image_name": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["image_name", "content"]
                        }
                    }
                },
                "required": ["extracted_texts"]
            }
        # Step 4: Map the extracted text to JSON using the predefined schema via an LLM call.
        final_json = self._convert_text_to_json(image_text_json, predefined_schema)
        return final_json

    def _extract_images_from_pdf(
        self, pdf_path: str, output_folder: str, max_image_size: int = 1024
    ) -> List[Dict[str, str]]:
        """
        Extracts images from a PDF using pdfplumber, saves them as PNG files in the output folder,
        and returns a list of dictionaries containing file paths and base64 encoded images.
        """
        try:
            def clip_bbox(bbox, parent_bbox):
                x0, top, x1, bottom = bbox
                p_x0, p_top, p_x1, p_bottom = parent_bbox
                new_x0 = max(x0, p_x0)
                new_top = max(top, p_top)
                new_x1 = min(x1, p_x1)
                new_bottom = min(bottom, p_bottom)
                return (new_x0, new_top, new_x1, new_bottom)
            
            os.makedirs(output_folder, exist_ok=True)
            image_counter = 1
            extracted_images = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    parent_bbox = page.bbox
                    for image in page.images:
                        bbox = (image["x0"], image["top"], image["x1"], image["bottom"])
                        clipped_bbox = clip_bbox(bbox, parent_bbox)
                        try:
                            cropped_page = page.crop(clipped_bbox)
                            page_image = cropped_page.to_image(resolution=300, antialias=True)
                        except Exception as e:
                            print(f"Error extracting image on page {page.page_number}: {e}")
                            continue
                        object_id = image.get("object_id", image_counter)
                        image_filename = f"page_{page.page_number}_img_{object_id}.png"
                        file_path = os.path.join(output_folder, image_filename)
                        try:
                            page_image.save(file_path, format="PNG", quantize=False)
                            print(f"Saved image to {file_path}")
                            base64_encoded = self._resize_and_encode_image(file_path, max_image_size)
                            extracted_images.append({"file_path": file_path, "base64": base64_encoded})
                        except Exception as e:
                            print(f"Error saving image on page {page.page_number}: {e}")
                        image_counter += 1
            return extracted_images
        except Exception as e:
            print(f"Error in _extract_images_from_pdf: {e}")
            return []

    def _resize_and_encode_image(self, image_path: str, max_image_size: int = 1024) -> Optional[str]:
        """
        Loads an image, resizes it while maintaining aspect ratio, and converts it to base64 encoding.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width > height:
                    max_size = (max_image_size, int(max_image_size * (height / width)))
                else:
                    max_size = (int(max_image_size * (width / height)), max_image_size)
                img = img.resize(max_size, Image.LANCZOS)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()
                base64_encoded_image = base64.b64encode(img_bytes).decode("utf-8")
                return base64_encoded_image
        except Exception as e:
            print(f"Error in _resize_and_encode_image: {e}")
            return None


# Structured Prompts & Fine-Tuning:
# When invoking the vision model, use structured prompts or fine-tune the model to output strictly formatted JSON.
# Example: Include examples in the prompt that define the exact structure expected.
# Post-Processing and Sanitization:
# After extracting JSON, use post-processing techniques (e.g., regex or dedicated parsers) to correct common errors and enforce strict formatting.
# Controlled Parameters:
# Use lower temperature settings and provide few-shot examples to guide the LLM toward more consistent outputs.
# Example: Setting a temperature of 0 or near 0 helps ensure deterministic responses.
# Post-Generation Verification:
# After the LLM generates the schema or final JSON, run it through a strict JSON validator. If errors are found, either automatically prompt the LLM for corrections or flag the result for manual review.
# Iterative Refinement:
# If using an LLM to generate a schema, run the output through multiple iterations of refinement and validation. Provide clear examples and constraints in the prompt to steer the model toward the correct format.


    def _get_image_text_json(self, images_info: List[Dict[str, str]]) -> str:
        """
        Extracts text from each image using an LLM.
        Iterates over the list of images, calls the LLM for each image to extract text,
        and returns a JSON-formatted string containing the results.
        """
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        extracted_texts = []
        for img_info in images_info:
            file_path = img_info.get("file_path")
            base64_image = img_info.get("base64")
            image_name = os.path.basename(file_path)
            if not base64_image:
                extracted_texts.append({
                    "image_name": image_name,
                    "content": "Error: No base64 image data available."
                })
                continue
            image_data = f"data:image/jpeg;base64,{base64_image}"
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all textual details from the image."},
                                {"type": "image_url", "image_url": {"url": image_data}},
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                text = response.choices[0].message.content
            except Exception as e:
                text = f"Error processing image: {str(e)}"
            extracted_texts.append({
                "image_name": image_name,
                "content": text
            })
        output_json = {"extracted_texts": extracted_texts}
        return json.dumps(output_json)

    def _load_baseline_schema(self, file_path: str) -> dict:
        """
        Loads the baseline JSON schema from a file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _convert_text_to_json(self, extracted_text: str, schema: dict) -> dict:
        """
        Maps the predefined JSON schema with the extracted text.
        Builds a prompt instructing the LLM to convert the extracted text into a JSON object 
        that strictly adheres to the provided JSON schema, calls the LLM, and parses the response.
        """
        prompt = (
            "You are a JSON Response Generator. Your task is to convert the plain text which is extracted from the images into a JSON object by using the provided JSON schema. Ensure that the output adheres strictly according to the schema and contains only the JSON object mapped to the extracted text.\n\n"
            "Extracted Text:\n"
            f"{extracted_text}\n\n"
            "JSON Schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
        )
        json_response = self._call_llm(prompt)
        # print(json_response)
        try:
            converted_json = json.loads(json_response)
            # print("!!!!!!!!!!@@@@@@@@@@@@@@@@@@######################")
            return converted_json
        except json.JSONDecodeError as e:
            print("Failed to parse JSON from LLM response:", e)
            return {}
        
        # return {}

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM using OpenAI's API with the provided prompt.
        """
        try:
            # api_key = os.getenv("OPENAI_API_KEY", "your_default_api_key_here")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4500,
                top_p=0.5
            )
            resp = response.choices[0].message.content
            print(resp)
            return resp
        except Exception as e:
            print("Error during LLM call:", e)
            return ""