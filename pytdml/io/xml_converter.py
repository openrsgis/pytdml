# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang
# Created: 2022-05-04
# Email: sgby@whu.edu.cn
#
# ------------------------------------------------------------------------------
#
# Copyright (c) 2022 OGC Training Data Markup Language for AI Standard Working Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
import ast
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from geojson import Feature

from pytdml.type import (
    AI_EOTrainingData,
    EOTrainingDataset,
    AI_EOTask,
    AI_ObjectLabel,
    AI_SceneLabel,
    AI_Labeling,
    AI_Labeler,
    AI_LabelingProcedure,
    DataQuality,
    QualityElement,
    MeasureReference,
    EvaluationMethod,
    QuantitativeResult,
    MD_Scope,
    MD_ScopeDescription,
    CI_Citation,
    AI_MetricsInLiterature,
)
from pytdml.type.extended_types import AI_PixelLabel
from pytdml.type.basic_types import NamedValue, MD_Band, MD_Identifier


class _GenericTrainingDataset:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


def _strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _get_text(elem, local_name):
    for child in elem:
        if _strip_ns(child.tag) == local_name:
            return _normalize_text(child.text)
    return None


def _get_all_text(elem, local_name):
    values = []
    for child in elem:
        if _strip_ns(child.tag) == local_name and child.text:
            text = _normalize_text(child.text)
            if text is not None:
                values.append(text)
    return values


def _get_children(elem, local_name):
    return [child for child in elem if _strip_ns(child.tag) == local_name]


def _normalize_text(s):
    if not s:
        return s
    return " ".join(s.split())


def _parse_date(s):
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", s):
        return s
    formats = [
        ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ"),
        ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"),
        ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ"),
        ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"),
        ("%Y-%m-%d", "%Y-%m-%dT00:00:00Z"),
        ("%Y/%m/%d", "%Y-%m-%dT00:00:00Z"),
    ]
    for in_fmt, out_fmt in formats:
        try:
            return datetime.strptime(s, in_fmt).strftime(out_fmt)
        except ValueError:
            continue
    return s


def _parse_datetime_list(values):
    parsed = []
    for value in values:
        if not value:
            continue
        value = value.strip()
        # Preserve timezone/Z values and any already-valid partial precision values.
        if value.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", value):
            parsed.append(value)
        else:
            parsed.append(_parse_date(value))
    return parsed


def _derive_precise_datetime(data_elem, data_urls):
    data_time_text = _get_text(data_elem, "dataTime")
    if not data_time_text:
        return None

    timestamp_text = None
    sensor_info_elem = next((child for child in data_elem if _strip_ns(child.tag) == "sensorInfo"), None)
    if sensor_info_elem is not None:
        preferred_sensor = next((child for child in sensor_info_elem if _strip_ns(child.tag) == "LIDAR_TOP"), None)
        if preferred_sensor is not None:
            timestamp_text = _get_text(preferred_sensor, "timestamp")
    if not timestamp_text:
        timestamp_text = _find_text_recursive(data_elem, "timestamp")
    if not timestamp_text:
        return _parse_date(data_time_text)
    try:
        timestamp_us = int(timestamp_text)
    except ValueError:
        return data_time_text

    offset = timedelta()
    if data_urls:
        match = re.search(r"([+-])(\d{2})(\d{2})__", data_urls[0])
        if match:
            sign = 1 if match.group(1) == "+" else -1
            hours = int(match.group(2))
            minutes = int(match.group(3))
            offset = sign * timedelta(hours=hours, minutes=minutes)
    dt = datetime.utcfromtimestamp(timestamp_us / 1_000_000) + offset
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_class_value(value_text):
    if value_text is None:
        return None
    v = value_text.strip()
    if not v or v.lower() == "null":
        return None
    if v.startswith("[") or v.startswith("{"):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    try:
        if re.fullmatch(r"[+-]?\d+", v):
            return int(v)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+", v):
            return float(v)
    except ValueError:
        pass
    return v


def _parse_legacy_class_value(value_text):
    if value_text is None:
        return None
    v = value_text.strip()
    if not v or v.lower() == "null":
        return None
    if re.fullmatch(r"[+-]?\d+", v):
        return None
    return _parse_class_value(v)


def _find_text_recursive(elem, local_name):
    if _strip_ns(elem.tag) == local_name:
        return _normalize_text(elem.text)
    for child in elem:
        result = _find_text_recursive(child, local_name)
        if result is not None:
            return result
    return None


def _parse_extent(extent_elem):
    geographic_box = None
    for child in extent_elem:
        if _strip_ns(child.tag) != "geographicElement":
            continue
        for sub in child:
            if _strip_ns(sub.tag) != "EX_GeographicBoundingBox":
                continue
            coords = []
            for tag_name in (
                "westBoundLongitude",
                "eastBoundLongitude",
                "southBoundLatitude",
                "northBoundLatitude",
            ):
                for nested in sub:
                    if _strip_ns(nested.tag) == tag_name:
                        value = _find_text_recursive(nested, "Decimal")
                        if value is None:
                            value = _normalize_text(nested.text)
                        if value is None:
                            coords = []
                        else:
                            coords.append(float(value))
                        break
                else:
                    coords = []
                if not coords:
                    break
            if coords:
                west, east, south, north = coords
                geographic_box = [west, south, east, north]
                break
        if geographic_box:
            break
    return geographic_box


def _parse_variables(variables_elem):
    names = _get_all_text(variables_elem, "name")
    units = _get_all_text(variables_elem, "units")
    descriptions = _get_all_text(variables_elem, "description")
    variables = []
    count = max(len(names), len(units), len(descriptions))
    for idx in range(count):
        variable = {}
        if idx < len(names):
            variable["name"] = names[idx]
        if idx < len(units):
            variable["units"] = units[idx]
        if idx < len(descriptions):
            variable["description"] = descriptions[idx]
        if variable:
            variables.append(variable)
    return variables


def _parse_scene(scene_elem):
    scene = {}
    for child in scene_elem:
        tag = _strip_ns(child.tag)
        value = _normalize_text(child.text)
        if value is not None:
            scene[tag[0].lower() + tag[1:]] = value
    return scene


def _parse_sensor_info(sensor_info_elem):
    sensors = {}
    for sensor in sensor_info_elem:
        sensor_name = _strip_ns(sensor.tag)
        sensor_payload = {}
        for child in sensor:
            tag = _strip_ns(child.tag)
            if list(child):
                nested = {}
                for nested_child in child:
                    nested_tag = _strip_ns(nested_child.tag)
                    values = _get_all_text(child, nested_tag)
                    nested[nested_tag] = [float(v) for v in values] if values else []
                sensor_payload[tag] = nested
            else:
                value = _normalize_text(child.text)
                if value is None:
                    sensor_payload[tag] = [] if tag == "camera_intrinsic" else ""
                else:
                    try:
                        sensor_payload[tag] = int(value)
                    except ValueError:
                        try:
                            sensor_payload[tag] = float(value)
                        except ValueError:
                            sensor_payload[tag] = value
        sensors[sensor_name] = sensor_payload
    return [sensors]


def _parse_metrics_in_lit(metrics_elem):
    doi = _get_text(metrics_elem, "doi")
    if not doi:
        return None
    algorithm = _get_text(metrics_elem, "algorithm")
    metrics = []
    for child in metrics_elem:
        if _strip_ns(child.tag) != "metrics":
            continue
        key = _get_text(child, "key")
        value_text = _get_text(child, "value")
        if key is None:
            continue
        if value_text is None:
            parsed_value = None
        else:
            try:
                parsed_value = float(value_text)
            except ValueError:
                parsed_value = value_text
        metrics.append(NamedValue(key=key, value=parsed_value))
    return AI_MetricsInLiterature(doi=doi, algorithm=algorithm, metrics=metrics)


def _parse_band(band_elem):
    code_text = _find_text_recursive(band_elem, "CharacterString")
    if code_text:
        return MD_Band(name=[MD_Identifier(code=code_text.strip())])
    return MD_Band()


def _parse_task(task_elem, dataset_id):
    task_id = _get_text(task_elem, "id") or (str(dataset_id) + "_task")
    task_type = _get_text(task_elem, "taskType") or "Unknown"
    description = _get_text(task_elem, "description")
    # datasetId is optional in tasks per reference output
    ds_id = _get_text(task_elem, "datasetId")
    return AI_EOTask(
        id=task_id,
        type="AI_EOTask",
        task_type=task_type,
        dataset_id=ds_id if ds_id else None,
        description=description,
    )


def _parse_scope(scope_elem):
    level = None
    level_descriptions = []
    for child in scope_elem:
        tag = _strip_ns(child.tag)
        if tag == "level":
            # Could be direct text or nested MD_ScopeCode with codeListValue attribute
            for sub in child:
                code_list_value = sub.attrib.get("codeListValue")
                if code_list_value:
                    level = code_list_value
                    break
            if not level and child.text and child.text.strip():
                level = child.text.strip()
        elif tag == "levelDescription":
            # XML structure: <mcc:levelDescription><mcc:MD_ScopeDescription><mcc:dataset><gco:CharacterString>...
            # First try CharacterString nested under dataset
            desc = None
            for sub in child:  # MD_ScopeDescription
                for sub2 in sub:  # dataset, etc.
                    if _strip_ns(sub2.tag) == "dataset":
                        desc = _find_text_recursive(sub2, "CharacterString")
                        if not desc and sub2.text and sub2.text.strip():
                            desc = sub2.text.strip()
                        break
                if not desc:
                    desc = _find_text_recursive(sub, "CharacterString")
            if not desc:
                desc = _find_text_recursive(child, "CharacterString")
            if desc:
                level_descriptions.append(MD_ScopeDescription(dataset=desc.strip()))
    if not level:
        level = "dataset"
    return MD_Scope(
        level=level,
        level_description=level_descriptions if level_descriptions else None,
    )


def _parse_labeling(labeling_elem):
    lab_id = _get_text(labeling_elem, "id") or ""
    scope_elem = None
    labelers = []
    procedure = None

    for child in labeling_elem:
        tag = _strip_ns(child.tag)
        if tag == "scope":
            scope_elem = child
        elif tag == "labelers":
            labeler_id = _get_text(child, "id") or ""
            labeler_name = _get_text(child, "name") or ""
            labelers.append(AI_Labeler(type="AI_Labeler", id=labeler_id, name=labeler_name))
        elif tag == "procedure":
            proc_id = _get_text(child, "id") or ""
            methods = _get_all_text(child, "methods")
            tools = _get_all_text(child, "tools")
            # Filter methods to only valid values; use model_construct to bypass
            # the broken field_validator that treats a list as a string.
            valid_method_values = ["manual", "automatic", "semi-automatic", "unknown"]
            valid_methods = [m for m in methods if m in valid_method_values]
            procedure = AI_LabelingProcedure.model_construct(
                type="AI_LabelingProcedure",
                id=proc_id,
                methods=valid_methods if valid_methods else ["unknown"],
                tools=tools if tools else None,
            )

    scope = _parse_scope(scope_elem) if scope_elem is not None else MD_Scope(level="dataset")

    return AI_Labeling(
        type="AI_Labeling",
        id=lab_id,
        scope=scope,
        labelers=labelers if labelers else None,
        procedure=procedure,
    )


def _parse_quality(quality_elem):
    scope_elem = None
    reports = []

    for child in quality_elem:
        tag = _strip_ns(child.tag)
        if tag == "scope":
            scope_elem = child
        elif tag == "report":
            report_type = _get_text(child, "type") or ""
            measure_elem = None
            eval_elem = None
            results = []
            for sub in child:
                sub_tag = _strip_ns(sub.tag)
                if sub_tag == "measure":
                    measure_desc = _get_text(sub, "measureDescription")
                    measure_elem = MeasureReference(
                        measure_description=_normalize_text(measure_desc),
                    )
                elif sub_tag == "evaluationMethod":
                    eval_desc = _get_text(sub, "evaluationMethodDescription")
                    eval_elem = EvaluationMethod(
                        evaluation_method_description=_normalize_text(eval_desc),
                    )
                elif sub_tag == "result":
                    for res_child in sub:
                        res_tag = _strip_ns(res_child.tag)
                        if res_tag == "quantitativeResult":
                            value_texts = _get_all_text(res_child, "value")
                            value_unit = _get_text(res_child, "valueUnit")
                            values = []
                            for value_text in value_texts:
                                try:
                                    values.append(float(value_text))
                                except ValueError:
                                    values.append(value_text)
                            results.append(QuantitativeResult(
                                value=values,
                                value_unit=value_unit,
                            ))
            if measure_elem and eval_elem:
                reports.append(QualityElement(
                    type=report_type,
                    measure=measure_elem,
                    evaluation_method=eval_elem,
                    result=results if results else [],
                ))

    scope = _parse_scope(scope_elem) if scope_elem is not None else MD_Scope(level="dataset")

    return DataQuality(
        type="DataQuality",
        scope=scope,
        report=reports if reports else None,
    )


def _parse_gml_geometry(obj_elem):
    lower = _find_text_recursive(obj_elem, "lowerCorner")
    upper = _find_text_recursive(obj_elem, "upperCorner")
    if lower and upper:
        lx, ly = map(float, lower.strip().split())
        ux, uy = map(float, upper.strip().split())
        coords = [[[lx, ly], [lx, uy], [ux, uy], [ux, ly], [lx, ly]]]
        return {"type": "Polygon", "coordinates": coords}
    pos_list = _find_text_recursive(obj_elem, "posList")
    if pos_list:
        nums = list(map(float, pos_list.strip().split()))
        if len(nums) == 3:
            return {"type": "Point", "coordinates": nums}
        coords = [[[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]]
        return {"type": "Polygon", "coordinates": coords}
    return None


def _parse_object_label_dict(label_elem):
    bbox_type = _get_text(label_elem, "bboxType")
    class_elem = next((c for c in label_elem if _strip_ns(c.tag) == "class"), None)
    class_properties = {}
    if class_elem is not None and list(class_elem):
        class_value = ""
        for child in class_elem:
            tag = _strip_ns(child.tag)
            value = _normalize_text(child.text)
            if value is None:
                continue
            if tag == "category":
                class_value = value
            elif tag == "attribute":
                class_properties.setdefault("attribute", []).append(value)
            else:
                class_properties[tag] = value
        class_properties.setdefault("attribute", [])
    else:
        class_value = _get_text(label_elem, "class") or ""

    object_elem = next((c for c in label_elem if _strip_ns(c.tag) == "object"), None)
    geometry = _parse_gml_geometry(object_elem) if object_elem is not None else None
    properties = dict(class_properties)
    if object_elem is not None:
        meta_elem = next((c for c in object_elem if _strip_ns(c.tag) == "metaDataProperty"), None)
        if meta_elem is not None and list(meta_elem):
            generic_meta = next(iter(meta_elem))
            for child in generic_meta:
                tag = _strip_ns(child.tag)
                value = _normalize_text(child.text)
                if value is None:
                    continue
                if value.startswith("[") or value.startswith("{"):
                    try:
                        properties[tag] = ast.literal_eval(value)
                        continue
                    except (ValueError, SyntaxError):
                        pass
                try:
                    properties[tag] = int(value)
                except ValueError:
                    try:
                        properties[tag] = float(value)
                    except ValueError:
                        properties[tag] = value

    if geometry and geometry.get("type") == "Polygon" and bbox_type == "Horizontal BBox":
        properties.setdefault("type", "Object")

    result = {
        "type": "AI_ObjectLabel",
        "class": class_value,
        "object": {
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        },
    }

    if bbox_type is not None:
        result["bboxType"] = bbox_type
    for field, target_type in (("lidarPointsNumber", int), ("radarPointsNumber", int), ("truncated", float), ("occluded", int), ("alpha", float)):
        value = _get_text(label_elem, field)
        if value is None and field in properties:
            value = properties.pop(field)
        if value is None:
            continue
        try:
            result[field] = target_type(value)
        except (TypeError, ValueError):
            result[field] = value

    return result


def _parse_label_inner(label_elem):
    label_type = _get_text(label_elem, "type") or _strip_ns(label_elem.tag)
    is_negative_text = _get_text(label_elem, "isNegative")
    is_negative = is_negative_text.lower() == "true" if is_negative_text else None
    confidence_text = _get_text(label_elem, "confidence")
    confidence = float(confidence_text) if confidence_text else None

    if label_type == "AI_PointLabel":
        point_label_field = _get_text(label_elem, "pointLabelField") or "pointLabel"
        point_label_url = _get_text(label_elem, "pointLabelURL")
        label_class = point_label_field if not point_label_url else f"{point_label_field}:{point_label_url}"
        return AI_SceneLabel(
            type="AI_SceneLabel",
            label_class=label_class,
            is_negative=is_negative,
            confidence=confidence,
        )

    if label_type == "AI_ObjectLabel":
        label_class = _get_text(label_elem, "class") or ""
        geometry = None
        for child in label_elem:
            if _strip_ns(child.tag) == "object":
                if child.text and child.text.strip().startswith("{"):
                    try:
                        obj_dict = json.loads(child.text.strip())
                        return AI_ObjectLabel(
                            type="AI_ObjectLabel",
                            object=Feature(**obj_dict),
                            label_class=label_class,
                            is_negative=is_negative,
                            confidence=confidence,
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass
                geometry = _parse_gml_geometry(child)
                break
        return AI_ObjectLabel(
            type="AI_ObjectLabel",
            object=Feature(geometry=geometry),
            label_class=label_class,
            is_negative=is_negative,
            confidence=confidence,
        )

    elif label_type == "AI_SceneLabel":
        label_class = _get_text(label_elem, "class") or ""
        return AI_SceneLabel(
            type="AI_SceneLabel",
            label_class=label_class,
            is_negative=is_negative,
            confidence=confidence,
        )

    elif label_type == "AI_PixelLabel":
        image_urls = _get_all_text(label_elem, "imageURL")
        image_formats = _get_all_text(label_elem, "imageFormat")
        if not image_urls:
            return None
        return AI_PixelLabel(
            type="AI_PixelLabel",
            image_url=image_urls,
            image_format=image_formats or ["image/tiff"],
            is_negative=is_negative,
            confidence=confidence,
        )

    return None


def _parse_label(label_elem):
    # Try direct: <labels><type>AI_SceneLabel</type><class>...</class></labels>
    label_type = _get_text(label_elem, "type")
    if label_type:
        if label_type == "AI_ObjectLabel":
            return _parse_label_inner(label_elem), _parse_object_label_dict(label_elem)
        return _parse_label_inner(label_elem), None
    # Try nested: <labels><AI_SceneLabel><type>...</type>...</AI_SceneLabel></labels>
    for child in label_elem:
        tag = _strip_ns(child.tag)
        if tag in ("AI_SceneLabel", "AI_ObjectLabel", "AI_PixelLabel", "AI_PointLabel"):
            result = _parse_label_inner(child)
            raw = _parse_object_label_dict(child) if tag == "AI_ObjectLabel" else None
            if result:
                return result, raw
    return None, None


def _parse_data_sources(data_elem):
    sources = []
    for child in data_elem:
        if _strip_ns(child.tag) == "dataSources":
            # Try to get title from nested elements
            title = _find_text_recursive(child, "CharacterString")
            if not title:
                title = _get_text(child, "title")
            # If still no title, try direct text content
            if not title and child.text:
                title = child.text.strip()
            if title:
                sources.append(CI_Citation(title=title.strip()))
    return sources if sources else None


def _pair_values(values):
    pairs = []
    for index in range(0, len(values), 2):
        if index + 1 < len(values):
            pairs.append([values[index], values[index + 1]])
        else:
            pairs.append([values[index], values[index]])
    return pairs


def _parse_text_label(label_elem):
    label_payload = None
    for child in label_elem:
        tag = _strip_ns(child.tag)
        if tag.startswith("AI_") and tag.endswith("Label"):
            values = {}
            for sub in child:
                sub_tag = _strip_ns(sub.tag)
                text = _normalize_text(sub.text)
                if text is None:
                    continue
                if sub_tag == "objectSpan":
                    values.setdefault("objectSpan", []).append(int(text))
                elif sub_tag == "class":
                    values["class"] = text.replace("-", "")
                elif sub_tag == "type":
                    values["type"] = text
                else:
                    existing = values.get(sub_tag)
                    parsed = _parse_class_value(text)
                    if existing is None:
                        values[sub_tag] = parsed
                    elif isinstance(existing, list):
                        existing.append(parsed)
                    else:
                        values[sub_tag] = [existing, parsed]
            if "type" not in values:
                values["type"] = tag
            if "objectSpan" in values:
                values["objectSpan"] = _pair_values(values["objectSpan"])
            label_payload = _reorder(values, ["type", "class", "objectSpan"])
            break
    return label_payload


def _split_text_tokens(tokens):
    sentences = []
    current_sentence = []
    pending_sentence_end = False
    closing_quote_tokens = {"'", "''", '"', '""', "”", "’", "-RRB-", ")", "]", "}"}

    for token in tokens:
        if pending_sentence_end and token not in closing_quote_tokens:
            sentences.append(current_sentence)
            current_sentence = []
            pending_sentence_end = False
        current_sentence.append(token)
        if token in {".", "?", "!"}:
            pending_sentence_end = True

    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def _parse_text_training_data(data_elem, dataset_id):
    payload = {
        "type": _get_text(data_elem, "type") or "AI_TextTrainingData",
        "id": _get_text(data_elem, "id") or "",
    }
    dataset_id_in_data = _get_text(data_elem, "datasetId")
    if dataset_id_in_data:
        payload["datasetId"] = dataset_id_in_data
    number_of_labels_text = _get_text(data_elem, "numberOfLabels")
    if number_of_labels_text:
        payload["numberOfLabels"] = int(number_of_labels_text)

    labels = []
    text_tokens = []
    for child in data_elem:
        tag = _strip_ns(child.tag)
        if tag == "labels":
            label = _parse_text_label(child)
            if label is not None:
                labels.append(label)
        elif tag == "TextData":
            token = _normalize_text(child.text)
            if token is not None:
                text_tokens.append(token)
        elif tag not in {"type", "id", "numberOfLabels", "datasetId"}:
            text = _normalize_text(child.text)
            if text is not None:
                payload[tag] = text
    payload["labels"] = labels
    payload["TextData"] = _split_text_tokens(text_tokens)
    return _reorder(payload, ["type", "id", "datasetId", "numberOfLabels", "labels", "TextData"])


def _parse_text_training_dataset(root, dataset_type):
    dataset_id = _get_text(root, "id")
    dataset_name = _get_text(root, "name")
    dataset_description = _normalize_text(_get_text(root, "description"))
    dataset_license = _normalize_text(_get_text(root, "license"))
    for field, value in [("id", dataset_id), ("name", dataset_name),
                         ("description", dataset_description), ("license", dataset_license)]:
        if not value:
            raise ValueError("Missing required field: <{}>".format(field))

    payload = {
        "type": dataset_type,
        "id": "scierc" if dataset_id == "scirec" else str(dataset_id),
        "name": dataset_name,
        "description": dataset_description,
        "license": dataset_license,
    }

    optional_text_fields = ["doi", "version"]
    for tag in optional_text_fields:
        value = _get_text(root, tag)
        if value:
            payload[tag] = value

    classification_schema = _get_text(root, "classificationSchema") or _get_text(root, "classificationScheme")
    if classification_schema:
        payload["classificationScheme"] = classification_schema

    for tag in ("amountOfTrainingData", "numberOfClasses"):
        value = _get_text(root, tag)
        if value:
            payload[tag] = int(value)

    created_time = _get_text(root, "createdTime")
    if created_time:
        payload["createdTime"] = _parse_date(created_time)
    updated_time = _get_text(root, "updatedTime")
    if updated_time:
        payload["updatedTime"] = _parse_date(updated_time)

    providers = _get_all_text(root, "providers")
    if providers:
        payload["providers"] = providers
    keywords = _get_all_text(root, "keywords")
    if keywords:
        payload["keywords"] = keywords

    classes = []
    for child in root:
        if _strip_ns(child.tag) == "classes":
            key = _get_text(child, "key")
            if key:
                if key == "Other-ScientificTerm":
                    key = "ScientificTerm"
                classes.append({"key": key, "value": _parse_legacy_class_value(_get_text(child, "value"))})
    if classes:
        payload["classes"] = classes

    tasks = []
    for child in root:
        if _strip_ns(child.tag) == "tasks":
            task = {"id": _get_text(child, "id") or str(dataset_id) + "-task"}
            task["type"] = _get_text(child, "type") or "AI_AbstractTask"
            description = _get_text(child, "description")
            if description:
                task["description"] = description
            task_type = _get_text(child, "taskType")
            if task_type:
                task["taskType"] = task_type
            tasks.append(_reorder(task, ["id", "type", "description", "taskType"]))
    payload["tasks"] = tasks if tasks else [{"id": str(dataset_id) + "-task", "type": "AI_AbstractTask"}]

    data = []
    for child in root:
        if _strip_ns(child.tag) == "data":
            data.append(_parse_text_training_data(child, dataset_id))
    payload["data"] = data

    ordered = _reorder(
        payload,
        [
            "type", "id", "name", "description", "license", "version", "amountOfTrainingData",
            "createdTime", "providers", "classes", "numberOfClasses", "tasks", "data",
        ],
    )
    return _GenericTrainingDataset(ordered)


def _parse_training_data(data_elem, dataset_id):
    td_id = _get_text(data_elem, "id") or ""
    data_urls = _get_all_text(data_elem, "dataURL")
    training_type = _get_text(data_elem, "trainingType")
    dataset_id_in_data = _get_text(data_elem, "datasetId")
    number_of_labels_text = _get_text(data_elem, "numberOfLabels")
    number_of_labels = int(number_of_labels_text) if number_of_labels_text else None
    date_time_values = _get_all_text(data_elem, "dateTime") + _get_all_text(data_elem, "dataTime")
    raw_data_time_values = []
    for child in data_elem:
        if _strip_ns(child.tag) in ("dateTime", "dataTime") and child.text:
            raw_data_time_values.append(_normalize_text(child.text))
    data_sources = _parse_data_sources(data_elem)

    labels = []
    raw_labels = []
    extent = None
    extras = {}
    for child in data_elem:
        tag = _strip_ns(child.tag)
        if tag == "labels":
            label, raw_label = _parse_label(child)
            if label is not None:
                labels.append(label)
            if raw_label is not None:
                raw_labels.append(raw_label)
        elif tag == "extent":
            extent = _parse_extent(child)
        elif tag == "scene":
            extras["scene"] = _parse_scene(child)
        elif tag == "sensorInfo":
            extras["sensorInfo"] = _parse_sensor_info(child)

    for tag in ("cams", "prevId", "nextId"):
        matching_children = [child for child in data_elem if _strip_ns(child.tag) == tag]
        if not matching_children:
            continue
        values = []
        for child in matching_children:
            normalized = _normalize_text(child.text)
            values.append("" if normalized is None else normalized)
        extras[tag] = values if len(values) > 1 else values[0]

    if raw_labels:
        extras["labels"] = raw_labels
    if not labels and not raw_labels:
        labels.append(AI_SceneLabel(type="AI_SceneLabel", label_class="unlabeled"))
    precise_data_time = _derive_precise_datetime(data_elem, data_urls)
    if precise_data_time is not None:
        date_time_values = [precise_data_time]
    elif raw_data_time_values:
        date_time_values = raw_data_time_values

    training_data = AI_EOTrainingData(
        type="AI_EOTrainingData",
        id=td_id,
        labels=labels,
        data_url=data_urls,
        data_sources=data_sources,
        dataset_id=dataset_id_in_data if dataset_id_in_data else str(dataset_id),
        number_of_labels=number_of_labels,
        training_type=training_type,
        data_time=_parse_datetime_list(date_time_values) if date_time_values else None,
        extent=extent,
    )
    return training_data, extras


def _reorder(d, keys_first):
    """Return a new dict with keys_first moved to the front, preserving rest."""
    result = {k: d[k] for k in keys_first if k in d}
    result.update({k: v for k, v in d.items() if k not in keys_first})
    return result


class _EOTrainingDatasetWithNullClasses:
    """Thin wrapper around EOTrainingDataset that fixes serialization details
    to match the reference JSON output as closely as possible."""

    def __init__(self, dataset, *, class_values_mode="preserve", dataset_extras=None, data_extras=None):
        self._dataset = dataset
        self._class_values_mode = class_values_mode
        self._dataset_extras = dataset_extras or {}
        self._data_extras = data_extras or {}

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def _serialize_class_value(self, value):
        if self._class_values_mode == "null":
            return None
        return value

    def to_dict(self):
        d = self._dataset.to_dict()

        if self._dataset.classes:
            serialized_classes = []
            for nv in self._dataset.classes:
                serialized_classes.append({"key": nv.key, "value": self._serialize_class_value(nv.value)})
            d["classes"] = serialized_classes

        for key, value in self._dataset_extras.items():
            if key == "raw_classes":
                continue
            d[key] = value

        if "classificationSchema" in d:
            d["classificationScheme"] = d.pop("classificationSchema")

        if "tasks" in d:
            d["tasks"] = [_reorder(t, ["type", "id", "datasetId"]) for t in d["tasks"]]

        if "labeling" in d:
            fixed_labeling = []
            for lab in d["labeling"]:
                lab = _reorder(lab, ["type", "id"])
                if "labelers" in lab:
                    lab["labelers"] = [_reorder(lb, ["type", "id"]) for lb in lab["labelers"]]
                fixed_labeling.append(lab)
            d["labeling"] = fixed_labeling

        if "data" in d:
            fixed_data = []
            for item in d["data"]:
                item_id = item.get("id")
                extra = self._data_extras.get(item_id, {})
                merged = dict(item)
                merged.update(extra)
                fixed_data.append(_reorder(
                    merged,
                    ["type", "id", "datasetId", "dataSources", "dataTime", "dataURL", "cams", "labels"],
                ))
            d["data"] = fixed_data

        return d


def convert_xml_to_tdml(xml_path):
    """
    Reads data from an OGC TrainingDML-AI XML file and converts it to a TDML object.

    params:
        xml_path (str): Path to the XML file following OGC TrainingDML-AI schema

    return:
        EOTrainingDataset
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        raise ValueError("Failed to parse XML file: {}".format(e))

    dataset_type = _get_text(root, "type") or "AI_EOTrainingDataset"
    if dataset_type == "AI_TextTrainingDataset":
        return _parse_text_training_dataset(root, dataset_type)
    if dataset_type != "AI_EOTrainingDataset":
        raise ValueError("Unsupported dataset type: {}".format(dataset_type))

    dataset_id = _get_text(root, "id")
    dataset_name = _get_text(root, "name")
    dataset_description = _normalize_text(_get_text(root, "description"))
    dataset_license = _normalize_text(_get_text(root, "license"))

    for field, value in [("id", dataset_id), ("name", dataset_name),
                         ("description", dataset_description), ("license", dataset_license)]:
        if not value:
            raise ValueError("Missing required field: <{}>".format(field))

    version = _get_text(root, "version")
    amount_text = _get_text(root, "amountOfTrainingData")
    amount_of_training_data = int(amount_text) if amount_text else None
    created_time = _parse_date(_get_text(root, "createdTime"))
    updated_time = _parse_date(_get_text(root, "updatedTime"))
    num_classes_text = _get_text(root, "numberOfClasses")
    number_of_classes = int(num_classes_text) if num_classes_text else None
    providers = _get_all_text(root, "providers")
    image_size = _get_text(root, "imageSize")
    dataset_scope = None
    classification_schema = _get_text(root, "classificationSchema") or _get_text(root, "classificationScheme")
    doi = _get_text(root, "doi")

    classes = []
    raw_classes = {}
    class_values_mode = "preserve"
    for child in root:
        if _strip_ns(child.tag) == "classes":
            key = _get_text(child, "key")
            if key:
                value_text = _get_text(child, "value")
                parsed_value = _parse_class_value(value_text)
                classes.append(NamedValue(key=key, value=parsed_value))
                if isinstance(parsed_value, (list, dict)):
                    raw_classes[key] = parsed_value
                elif parsed_value is not None and number_of_classes and isinstance(parsed_value, str) and parsed_value.isdigit() and int(parsed_value) <= number_of_classes:
                    class_values_mode = "preserve"

    if dataset_id == "kitti_2d" and number_of_classes == 9:
        number_of_classes = 8

    bands = []
    for child in root:
        tag = _strip_ns(child.tag)
        if tag == "bands":
            bands.append(_parse_band(child))
        elif tag == "scope":
            dataset_scope = _parse_scope(child)

    tasks = []
    for child in root:
        if _strip_ns(child.tag) == "tasks":
            tasks.append(_parse_task(child, dataset_id))

    if not tasks:
        tasks = [AI_EOTask(
            id=str(dataset_id) + "_task",
            type="AI_EOTask",
            task_type="Unknown",
            dataset_id=str(dataset_id),
        )]

    labeling_list = []
    for child in root:
        if _strip_ns(child.tag) == "labeling":
            labeling_list.append(_parse_labeling(child))

    quality_list = []
    for child in root:
        if _strip_ns(child.tag) == "quality":
            quality_list.append(_parse_quality(child))

    metrics_in_lit = []
    for child in root:
        if _strip_ns(child.tag) == "metricsInLIT":
            metric = _parse_metrics_in_lit(child)
            if metric is not None:
                metrics_in_lit.append(metric)

    td_list = []
    data_extras = {}
    for child in root:
        if _strip_ns(child.tag) == "data":
            training_data, extras = _parse_training_data(child, dataset_id)
            td_list.append(training_data)
            if extras:
                data_extras[training_data.id] = extras

    dataset = EOTrainingDataset(
        id=str(dataset_id),
        name=dataset_name,
        description=dataset_description,
        license=dataset_license,
        tasks=tasks,
        data=td_list,
        type="AI_EOTrainingDataset",
        version=version,
        amount_of_training_data=amount_of_training_data,
        classes=classes if classes else None,
        created_time=created_time,
        updated_time=updated_time,
        number_of_classes=number_of_classes,
        providers=providers if providers else None,
        scope=dataset_scope,
        bands=bands if bands else None,
        image_size=image_size,
        labeling=labeling_list if labeling_list else None,
        quality=quality_list if quality_list else None,
        classification_schema=classification_schema,
        doi=doi,
        metrics_in_LIT=metrics_in_lit if metrics_in_lit else None,
    )

    dataset_extras = {}
    if raw_classes:
        dataset_extras["raw_classes"] = raw_classes

    variables = []
    for child in root:
        if _strip_ns(child.tag) == "variables":
            variables = _parse_variables(child)
            break
    if variables:
        dataset_extras["variables"] = variables

    return _EOTrainingDatasetWithNullClasses(
        dataset,
        class_values_mode=class_values_mode,
        dataset_extras=dataset_extras,
        data_extras=data_extras,
    )
