import jsonschema
import requests

# Define the URL of the remote JSON schema
remote_schema_url = "https://raw.githubusercontent.com/opengeospatial/TrainingDML-AI_SWG/main/schemas/1.0/json_schema/ai_pixelLabel.json"

# Load the remote schema from the URL
response = requests.get(remote_schema_url)
remote_schema = response.json()

json_data = {
    "type":"AI_PixelLabel",
    "imageFormat": ["png"],
    "imageURL":["this_is_not_a_url.tif"]
}

# Validate the JSON data against the remote schema
try:
    jsonschema.validate(instance=json_data, schema=remote_schema)
    print("JSON data is valid according to the remote schema.")
except jsonschema.exceptions.ValidationError as e:
    print(f"JSON data is not valid: {e}")

# json_data = {
#     # "confidence":1,
#     # "isNegative": False,
#     "type":"AI_ObjectLabel",
#     "class": "ship",
#     "object": {
#     "type": "Feature",
#     "geometry": {
#       "type": "Polygon",
#       "coordinates": [
#         [
#           2306.0,
#           729.0
#         ],
#         [
#           2330.0,
#           729.0
#         ],
#         [
#           2330.0,
#           744.0
#         ],
#         [
#           2306.0,
#           744.0
#         ],
#         [
#           2306.0,
#           729.0
#         ]
#       ]
#     }
#     },
#     "bboxType": "Horizontal BBox",
# }
# json_data = {
#     "type":"AI_SceneLabel",
#     "class": "industrial",
# }

# # json_data = {
# #     "confidence":1,
# #     "isNegative": True,
# #     "type":"AI_ObjectLabel",
# #     "imageFormat": ["png"],
# #     "imageURL":["this_is_not_a_url.tif"]
# # }

# # print(SceneLabel.schema_json(indent=2))
# print(SceneLabel(**json_data).json())