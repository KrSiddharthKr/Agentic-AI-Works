import os
import json
import pdfplumber
import openai
from openai import OpenAI
import jsonschema
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool

# Input schema for the PDF Table Extraction Tool
class PDFTableExtractionInput(BaseModel):
    pdf_path: str = Field(..., description="The file path of the PDF to extract table data from.")
    baseline_schema_path: str = Field(..., description="The file path of the predefined baseline JSON schema.")

class PDFTableExtractionTool(BaseTool):
    name: str = "PDF Table Extraction Tool"
    description: str = (
        "Extracts table data from a PDF file, loads a predefined JSON schema, "
        "prompts the LLM to map the extracted table data to the JSON schema, "
        "and validates the resulting JSON."
    )
    args_schema: type = PDFTableExtractionInput

    def _run(self, pdf_path: str, baseline_schema_path: str) -> dict:
        """
        Executes the workflow:
          1. Extracts table data from the PDF.
          2. Loads the predefined JSON schema.
          3. Prompts the LLM to map the extracted table data to the JSON schema.
          4. Validates the output JSON against the schema.
        
        Returns:
            A dictionary with the extracted tables data, the schema used, 
            the JSON output from the LLM, and the validation status.
        """
        # Step 1: Extract table data from PDF.
        tables_data = self._extract_tables_from_pdf(pdf_path)
        
        # Step 2: Load the predefined JSON schema.
        baseline_schema = self._load_baseline_schema(baseline_schema_path)
        
        # Step 3: Convert the extracted table data into a JSON object using the schema.
        converted_json = self._convert_tables_to_json(tables_data, json.dumps(baseline_schema))
        
        # Step 4: Validate the output JSON against the baseline schema.
        valid = False
        try:
            jsonschema.validate(converted_json, baseline_schema)
            valid = True
        except jsonschema.ValidationError as e:
            print("Validation error:", e)
        
        return {
            "tables_data": tables_data,
            "schema_used": baseline_schema,
            "converted_json": converted_json,
            "valid": valid
        }
    
    # ---------------------------
    # Private Helper Methods
    # ---------------------------
    
    def _extract_tables_from_pdf(self, pdf_path: str) -> list:
        """
        Extracts table data from a PDF file.
        
        Returns:
            A list of dictionaries, each containing the page number and the table data.
        """
        tables_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        tables_data.append({
                            "page": i + 1,
                            "table": table  # Each table is a list of lists representing rows and cells.
                        })
        # return json.dumps(tables_data, indent=4)
        return tables_data

    def _load_baseline_schema(self, file_path: str) -> dict:
        """
        Loads the predefined JSON schema from a file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM using OpenAI's API to convert the extracted table data into a JSON object.
        """
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Replace with your chosen model if needed.
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error during LLM call:", e)
            return ""
    
    def _convert_tables_to_json(self, tables_data: list, schema: json) -> json:
        """
        Converts the extracted table data into a JSON object that conforms to the provided JSON schema.
        
        Constructs a prompt that instructs the LLM to map the table data to a JSON object following the schema.
        """
        # prompt = (
        #     f'You are a helpful assistant who converts table text into JSON objects. You are provided with a JSON schema {schema} and the extracted table text {tables_data}. Your task is to read, understand, combine the extracted table text data from {tables_data} with the provided JSON schema object from {schema}. Note that the {tables_data} may include "null" values in the Stage column. Whenever you encounter a "null", replace it with the most recent non-"null" stage value from the rows above. For example, if a "null" appears after "Mobile & OTP Verification", you should replace that "null" with "Mobile & OTP Verification". Ensure your final JSON output strictly adheres to the provided schema and accurately reflects the transformed table data.'
        #     # f"{json.dumps(tables_data, indent=2)}\n\n"
        #     # "JSON Schema:\n"
        #     # f"{json.dumps(schema, indent=2)}\n\n"
        # )
        prompt=f""" Convert the follwing {tables_data} table data to JSON object."""
    
        json_response = self._call_llm(prompt)
        try:
            converted_json = json.loads(json_response)
            return converted_json
        except json.JSONDecodeError as e:
            print("Failed to parse JSON from LLM response:", e)
            return {}



#     data = [
#     {
#         "page": 15,
#         "table": [
#             [
#                 "Stage",
#                 "Scanerio",
#                 "Feedback"
#             ],
#             [
#                 "Mobile & OTP Verification",
#                 "Post-login session timeout happened",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "The customer entered 3 times invalid OTP.\nPlease try again after some time and the\nfeedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "PreApproved Offer",
#                 "customer post looking at the loan amount\nand declined the offer",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Session timeout happened, post that show\nfeedback message",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers have no offer to give redirection\nto the nearest branch and show feedback\nform",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "Personal & Loan Details",
#                 "Pan verification failed ps that show feedback\nmessage",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customer have no offer give redirection to\nnearest branch and show feedback form",
#                 "the feedback form will be\nshown"
#             ],
#             [
#                 "Employment Details",
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "the feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "the feedback form will be\nshown"
#             ],
#             [
#                 "Account Details",
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "Customer Verification",
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ]
#         ]
#     },
#     {
#         "page": 16,
#         "table": [
#             [
#                 "",
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "Esign",
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "ENach",
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "Disbursal",
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Post successful interaction, show feedback\nmsg",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 "Status Check",
#                 "In case of any tech failure, show a technical\nissue msg and feedback form",
#                 "feedback form will be\nshown"
#             ],
#             [
#                 None,
#                 "Customers dropped, show redirection to the\nnearest branch and show feedback form",
#                 "feedback form will be\nshown"
#             ]
#         ]
#     },
#     {
#         "page": 18,
#         "table": [
#             [
#                 "Stage",
#                 "Reversal possible",
#                 "Remark"
#             ],
#             [
#                 "Mobile verification",
#                 "-",
#                 ""
#             ],
#             [
#                 "Preapproved offer",
#                 "Mobile verification stage (change mobile number)",
#                 ""
#             ],
#             [
#                 "personal details & loan\ndetails",
#                 "-",
#                 ""
#             ],
#             [
#                 "Employment details",
#                 "-",
#                 ""
#             ],
#             [
#                 "Account details",
#                 "-",
#                 ""
#             ],
#             [
#                 "Customer Verification",
#                 "Preapproved offer (Can change Offer amount and Loan amount), Mobile\nverification stage (change mobile number), Employer name(can change\nemployer name), Account details( To add disbursement account)",
#                 ""
#             ]
#         ]
#     },
#     {
#         "page": 21,
#         "table": [
#             [
#                 "SL No",
#                 "Open Point",
#                 "Owner",
#                 "Status"
#             ],
#             [
#                 "1",
#                 "Discussion with kowri on 2 API Loan detail and loan\nstatus to be called before offer API",
#                 "kowri.jayara\nman@hdbfs.\ncom@bhavan\ni",
#                 "Open"
#             ],
#             [
#                 "2",
#                 "Google Account to be created by HDB and token details\nto be shared with trustt for TTS (text to speech feature)",
#                 "HDB",
#                 "Open"
#             ],
#             [
#                 "3",
#                 "Discussion on Map API provider for showing nearest\nbranch for drop-off cases",
#                 "HDB",
#                 "Open"
#             ]
#         ]
#     },
#     {
#         "page": 22,
#         "table": [
#             [
#                 "4",
#                 "SMS whitelisting is pending",
#                 "HDB",
#                 "Closed"
#             ]
#         ]
#     }
# ]
