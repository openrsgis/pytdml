import json

from pytdml.io import write_to_json, read_from_json
from pytdml.type import extended_types_old, extended_types, basic_types

_IMAGE_FORMATS = {
    "image/aces",
    "image/apng",
    "image/avci",
    "image/avcs",
    "image/avif",
    "image/bmp",
    "image/cgm",
    "image/dicom-rle",
    "image/dpx",
    "image/emf",
    "image/example",
    "image/fits",
    "image/g3fax",
    "image/gif",
    "image/heic",
    "image/heic-sequence",
    "image/heif",
    "image/heif-sequence",
    "image/hej2k",
    "image/hsj2",
    "image/ief",
    "image/j2c",
    "image/jls",
    "image/jp2",
    "image/jpeg",
    "image/jph",
    "image/jphc",
    "image/jpm",
    "image/jpx",
    "image/jxr",
    "image/jxrA",
    "image/jxrS",
    "image/jxs",
    "image/jxsc",
    "image/jxsi",
    "image/jxss",
    "image/ktx",
    "image/ktx2",
    "image/naplps",
    "image/png",
    "image/prs.btif",
    "image/prs.pti",
    "image/pwg-raster",
    "image/svg+xml",
    "image/t38",
    "image/tiff",
    "image/tiff-fx",
    "image/tiff; application=geotiff",
    "image/vnd.adobe.photoshop",
    "image/vnd.airzip.accelerator.azv",
    "image/vnd.cns.inf2",
    "image/vnd.dece.graphic",
    "image/vnd.djvu",
    "image/vnd.dwg",
    "image/vnd.dxf",
    "image/vnd.dvb.subtitle",
    "image/vnd.fastbidsheet",
    "image/vnd.fpx",
    "image/vnd.fst",
    "image/vnd.fujixerox.edmics-mmr",
    "image/vnd.fujixerox.edmics-rlc",
    "image/vnd.globalgraphics.pgb",
    "image/vnd.microsoft.icon",
    "image/vnd.mix",
    "image/vnd.ms-modi",
    "image/vnd.mozilla.apng",
    "image/vnd.net-fpx",
    "image/vnd.pco.b16",
    "image/vnd.radiance",
    "image/vnd.sealed.png",
    "image/vnd.sealedmedia.softseal.gif",
    "image/vnd.sealedmedia.softseal.jpg",
    "image/vnd.svf",
    "image/vnd.tencent.tap",
    "image/vnd.valve.source.texture",
    "image/vnd.wap.wbmp",
    "image/vnd.xiff",
    "image/vnd.zbrush.pcx",
    "image/webp",
    "image/wmf",
    "image/x-emf",
    "image/x-wmf",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/x-netcdf",
    "application/geopackage+sqlite3"
}


def validate_image_format(image_format):
    if not image_format:
        return None
    else:
        valid_formats = []
        item = image_format.strip().lower()
        if item in _IMAGE_FORMATS:
            valid_formats.append(item)
        else:
            if "/" not in item:
                candidate = f"image/{item}"
                if candidate in _IMAGE_FORMATS:
                    valid_formats.append(candidate)
                else:
                    return None
            else:
                return None
        return valid_formats


def version_converter(old_v_path):
    with open(old_v_path, "r", encoding='utf-8') as f:
        json_dict = json.load(f)
    old_v_ds = extended_types_old.EOTrainingDataset.from_dict(json_dict).to_dict()

    # tasks
    tasks = old_v_ds["tasks"]
    new_tasks = [extended_types.AI_EOTask(
        id=task["id"],
        type="AI_EOTask",
        task_type=task["taskType"],
        dataset_id=task["datasetId"],
        description=task["description"]
    ) for task in tasks]

    # data
    new_data = []
    for data in old_v_ds["data"]:
        # labels
        labels = data["labels"]
        new_labels = []
        if labels[0]["type"] == "SceneLabel":
            new_labels = [extended_types.AI_SceneLabel(
                type="AI_SceneLabel",
                is_negative=label["isNegative"],
                confidence=label["confidence"],
                label_class=label["class"]
            ) for label in labels]
        elif labels[0]["type"] == "ObjectLabel":
            new_labels = [extended_types.AI_ObjectLabel(
                type="AI_ObjectLabel",
                is_negative=label["isNegative"],
                confidence=label["confidence"],
                object=label.get("object", None),
                label_class=label["class"],
                date_time=label["dateTime"],
                bbox_type=label["bboxType"]
            ) for label in labels]
        elif labels[0]["type"] == "PixelLabel":
            new_labels = []
            for label in labels:
                if label["imageURL"] and isinstance(label["imageURL"], list):
                    image_url = label["imageURL"]
                elif label["imageURL"] and isinstance(label["imageURL"], str):
                    image_url = [label["imageURL"]]
                else:
                    image_url = None

                image_format = validate_image_format(label["imageFormat"])
                new_labels.append(extended_types.AI_PixelLabel(
                    type="AI_PixelLabel",
                    is_negative=label["isNegative"],
                    confidence=label["confidence"],
                    image_url=image_url,
                    image_format=image_format,
                ))
        # labeling
        if data["labeling"]:
            labelings = data["labeling"]
            new_labeling = [basic_types.AI_Labeling(
                id=labeling["id"],
                type="AI_Labeling",
                labelers=[basic_types.AI_Labeler(
                    id=labeler["id"],
                    name=labeler["name"],
                    type="AI_Labeler"
                ) for labeler in labeling["labelers"]] if labeling["labelers"] else None,
                procedure=basic_types.AI_LabelingProcedure(
                    type="AI_LabelingProcedure",
                    id=labeling["procedure"]["id"],
                    methods=labeling["procedure"]["methods"],
                    tools=labeling["procedure"]["tools"]
                ) if labeling["procedure"] else None,
            ) for labeling in labelings]
        else:
            new_labeling = None
        # quality
        if data["quality"]:
            quality = data["quality"]
            new_quality = basic_types.DataQuality(
                type="DataQuality",
                scope=basic_types.MD_Scope(
                    level=quality["scope"]["level"],
                    level_description=basic_types.MD_ScopeDescription(
                        attributes=" ".join(quality["scope"]["levelDescription"]["attributes"]),
                        features=" ".join(quality["scope"]["levelDescription"]["features"]),
                        feature_instances=" ".join(quality["scope"]["levelDescription"]["featureInstances"]),
                        attribute_instances=" ".join(quality["scope"]["levelDescription"]["attributeInstances"]),
                        dataset=quality["scope"]["levelDescription"]["dataset"],
                        other=quality["scope"]["levelDescription"]["other"]
                        ) if quality["scope"]["levelDescription"] else None
                ) if quality["scope"] else None
            )
        else:
            new_quality = None

        # data_url
        if data["dataURL"] and isinstance(data["dataURL"], str):
            data_url = [data["dataURL"]]
        elif data["dataURL"] and isinstance(data["dataURL"], list):
            data_url = data["dataURL"]
        else:
            data_url = None
        # data_time
        if data["dateTime"] and isinstance(data["dateTime"], str):
            data_time = [data["dateTime"]]
        elif data["dateTime"] and isinstance(data["dateTime"], list):
            data_time = data["dateTime"]
        else:
            data_time = None
        new_data.append(extended_types.AI_EOTrainingData(
            type="AI_EOTrainingData",
            id=data["id"],
            data_url=data_url,
            labels=new_labels,
            dataSet_id=data["datasetId"],
            data_sources=data["dataSources"],
            number_of_labels=data["numberOfLabels"],
            labeling=new_labeling,
            training_type=data["trainingType"],
            quality=new_quality,
            extent=data["extent"],
            data_time=data_time
        ))

    # data_sources
    if old_v_ds["dataSources"]:
        data_sources = old_v_ds["dataSources"]
        new_data_source = [basic_types.CI_Citation(
            title=data_source
        ) for data_source in data_sources]
    else:
        new_data_source = None

    # metrics_in_LIT
    if old_v_ds["metricsInLIT"]:
        metrics_in_literature = old_v_ds["metricsInLIT"]
        new_metrics_in_LIT = [basic_types.AI_MetricsInLiterature(
            doi=item["doi"],
            metrics=basic_types.NamedValue(
                key=item["metrics"]["key"],
                value=item["metrics"]["value"]
            ) if item["metrics"] else None,
            algorithm=item["algorithm"]
        ) for item in metrics_in_literature]
    else:
        new_metrics_in_LIT = None

    # classes
    if old_v_ds["classes"]:
        classes = old_v_ds["classes"]
        if isinstance(classes[0], dict):
            new_classes = [basic_types.NamedValue(
                key=next(iter(classes[i].items()))[0],
                value=next(iter(classes[i].items()))[1]
            ) for i in range(len(classes))]
        else:
            new_classes = [basic_types.NamedValue(
                key=classes[i],
                value=i + 1
            ) for i in range(len(classes))]
    else:
        new_classes = None

    # statistics_info
    if old_v_ds["statisticsInfo"]:
        statistics_info = old_v_ds["statisticsInfo"]
        new_info = [basic_types.NamedValue(
            key=next(iter(info.items()))[0],
            value=next(iter(info.items()))[1]
        ) for info in statistics_info]
    else:
        new_info = None

    # scope
    if old_v_ds["scope"]:
        scope = old_v_ds["scope"]
        new_scope = basic_types.MD_Scope(
            level=scope["level"],
            level_description=[basic_types.MD_ScopeDescription(
                attributes=" ".join(levelDescription["attributes"]),
                features=" ".join(levelDescription["features"]),
                feature_instances=" ".join(levelDescription["featureInstances"]) if levelDescription["featureInstances"] else None,
                attribute_instances=" ".join(levelDescription["attributeInstances"]) if levelDescription["attributeInstances"] else None,
                dataset=levelDescription["dataset"],
                other=levelDescription["other"]
            ) for levelDescription in scope["levelDescription"]] if scope["levelDescription"] else None
        )
    else:
        new_scope = None

    # labeling
    if old_v_ds["labeling"]:
        labelings = old_v_ds["labeling"]
        new_labeling = [basic_types.AI_Labeling(
            id=labeling["id"],
            type="AI_Labeling",
            labelers=[basic_types.AI_Labeler(
                id=labeler["id"],
                name=labeler["name"],
                type="AI_Labeler"
            ) for labeler in labeling["labelers"]] if labeling["labelers"] else None,
            procedure=basic_types.AI_LabelingProcedure(
                type="AI_LabelingProcedure",
                id=labeling["procedure"]["id"],
                methods=labeling["procedure"]["methods"],
                tools=labeling["procedure"]["tools"]
            ) if labeling["procedure"] else None,
        ) for labeling in labelings]
    else:
        new_labeling = None

    # quality
    if old_v_ds["quality"]:
        quality = old_v_ds["quality"]
        new_quality = basic_types.DataQuality(
            type="DataQuality",
            scope=basic_types.MD_Scope(
                level=quality["scope"]["level"],
                level_description=basic_types.MD_ScopeDescription(
                    attributes=" ".join(quality["scope"]["levelDescription"]["attributes"]),
                    features=" ".join(quality["scope"]["levelDescription"]["features"]),
                    feature_instances=" ".join(quality["scope"]["levelDescription"]["featureInstances"]),
                    attribute_instances=" ".join(quality["scope"]["levelDescription"]["attributeInstances"]),
                    dataset=quality["scope"]["levelDescription"]["dataset"],
                    other=quality["scope"]["levelDescription"]["other"]
                    ) if quality["scope"]["levelDescription"] else None
            ) if quality["scope"] else None
        )
    else:
        new_quality = None

    # changesets
    if old_v_ds["changesets"]:
        changesets = old_v_ds["changesets"]
        new_changesets = [basic_types.AI_TDChangeset(
            type="AI_TDChangeset",
            id=changeset["id"],
            change_count=changeset["changeCount"],
            dataset_id=changeset["datasetId"],
            version=changeset["version"],
            created_time=changeset["createdTime"]
        ) for changeset in changesets]
    else:
        new_changesets = None

    # bands
    if old_v_ds["bands"]:
        bands = old_v_ds["bands"]
        new_bands = [basic_types.MD_Band(
            name=[basic_types.MD_Identifier(
                code=band
            )]) for band in bands]
    else:
        new_bands = None

    new_v_ds = extended_types.EOTrainingDataset(
        id=old_v_ds["id"],
        name=old_v_ds["name"],
        description=old_v_ds["description"],
        license=old_v_ds["license"],
        tasks=new_tasks,
        data=new_data,
        type="AI_EOTrainingDataset",
        amount_of_training_data=len(new_data),
        classes=new_classes,
        classification_schema=old_v_ds["classificationSchema"],
        created_time=old_v_ds["createdTime"],
        data_sources=new_data_source,
        doi=old_v_ds["doi"],
        keywords=old_v_ds["keywords"],
        number_of_classes=old_v_ds["numberOfClasses"],
        providers=[old_v_ds["providers"]] if isinstance(old_v_ds["providers"], str) else old_v_ds["providers"],
        scope=new_scope,
        statistics_info=new_info,
        updated_time=old_v_ds["updatedTime"],
        version=old_v_ds["version"],
        labeling=new_labeling,
        metrics_in_LIT=new_metrics_in_LIT,
        quality=new_quality,
        changesets=new_changesets,
        bands=new_bands,
        extent=old_v_ds["extent"],
        image_size=old_v_ds["imageSize"]
    )

    return new_v_ds

if __name__ == '__main__':
    # old_v_path = "D:\\GroupAffairs\\TDML\\PyTDML\\datasetTDEncodes\\WHU-RS19_old.json"
    # new_v_ds = version_converter(old_v_path)
    # write_to_json(new_v_ds, "D:\\GroupAffairs\\TDML\\PyTDML\\datasetTDEncodes\\WHU-RS19.json")
    # ds = read_from_json("D:\\GroupAffairs\\TDML\\PyTDML\\datasetTDEncodes\\RSD46-WHU.json")
    # print(ds.to_dict().keys())
    classes = ["Grass", "Football Field", "Industrial Facilities", "Residential Land", "Harbor", "Farmland", "Parking Lot", "Park", "Store", "Beach", "Railway Station", "Pond", "Mountain", "Airport", "Woodland", "River", "Viaduct", "Desert", "Cross River Bridge"]
    new_classes = [basic_types.NamedValue(
        key=next(iter(classes[0].items()))[0],
        value=next(iter(classes[0].items()))[1]
    ) for i in range(len(classes))]
    print(new_classes)