import pytest
from pydantic import ValidationError, BaseModel, field_validator
import jsonschema
import requests

from pytdml.type.all_types import _validate_date, to_camel, AI_Labeler

base_url = "https://raw.githubusercontent.com/opengeospatial/TrainingDML-AI_SWG/main/schemas/1.0/json_schema/{}.json"


class test_date_model(BaseModel):
    date: str

    @field_validator("date")
    def validate_date(cls, v):
        return _validate_date(v)


# Test valid date-time format
def test_validate_datetime_format():
    valid_datetime = "2023-10-27T14:30:00"
    result = _validate_date(valid_datetime)
    assert result == valid_datetime


# Test valid date format
def test_validate_date_format():
    valid_date = "2023-10-27"
    result = _validate_date(valid_date)
    assert result == valid_date


# Test valid time format
def test_validate_time_format():
    valid_time = "14:30:00"
    result = _validate_date(valid_time)
    assert result == valid_time


# Test valid year format
def test_validate_year_format():
    valid_year = "2023"
    result = _validate_date(valid_year)
    assert result == valid_year


# Test valid year-month format
def test_validate_year_month_format():
    valid_year_month = "2023-10"
    result = _validate_date(valid_year_month)
    assert result == valid_year_month


# Test invalid date format
def test_invalid_format():
    invalid_date = "2023/10/27"
    data = {"date": invalid_date}
    with pytest.raises(
        ValidationError,
        match=rf"String {invalid_date} does not match any allowed format",
    ):
        test_date_model(**data)


# Test invalid date
def test_invalid_date():
    invalid_date = "2023-13-35"
    data = {"date": invalid_date}
    with pytest.raises(
        ValidationError,
        match=rf"String {invalid_date} does not match any allowed format",
    ):
        test_date_model(**data)


# Test invalid time
def test_invalid_time():
    invalid_time = "25:70:70"
    data = {"date": invalid_time}
    with pytest.raises(
        ValidationError,
        match=rf"String {invalid_time} does not match any allowed format",
    ):
        test_date_model(**data)


# Test invalid year
def test_invalid_year():
    invalid_year = "1800"
    data = {"date": invalid_year}
    with pytest.raises(
        ValidationError,
        match=rf"String {invalid_year} does not match any allowed format",
    ):
        test_date_model(**data)


# Test invalid year-month
def test_invalid_year_month():
    invalid_year_month = "2023-15"
    data = {"date": invalid_year_month}
    with pytest.raises(
        ValidationError,
        match=rf"String {invalid_year_month} does not match any allowed format",
    ):
        test_date_model(**data)


# Test to_camel
def test_to_camel():
    camel_string = "thisIsACamelString"
    snake_string = "this_is_a_camel_string"
    assert to_camel(snake_string) == camel_string


# Test invalid Labeler
def test_required_elements_with_Labeler():
    data = {
        "type": "AI_Labeler",
        "name": "zhaoyan"
    }
    with pytest.raises( ValidationError):
        AI_Labeler(**data)

# Test valid Labeler and with remote schema
def test_valid_Labeler_schema():
    data = {
        "type": "AI_Labeler",
        "id": "1",
        "name": "zhaoyan"
    }
    labeler = AI_Labeler(**data)

    remote_schema_url = base_url.format("ai_labeler")
    response = requests.get(remote_schema_url)
    remote_schema = response.json()

    jsonschema.validate(instance=labeler.model_dump(), schema=remote_schema)
