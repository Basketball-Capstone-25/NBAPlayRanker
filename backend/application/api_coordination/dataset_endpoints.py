"""Authenticated catalog and column definitions for loaded runtime datasets."""

from fastapi import APIRouter, Depends, HTTPException

from application.api_coordination.auth_dependency import require_auth
from infrastructure.data_access.dataset_metadata import DATASETS, dataset_catalog
from infrastructure.data_access.data_dictionary import dictionary_entries


def create_dataset_router(synergy_source):
    router = APIRouter(tags=["datasets"], dependencies=[Depends(require_auth)])

    @router.get("/meta/datasets")
    def datasets():
        return {"datasets": dataset_catalog(synergy=synergy_source())}

    @router.get("/meta/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        if dataset_id not in DATASETS:
            raise HTTPException(404, "Unknown dataset.")
        item = next(d for d in dataset_catalog(synergy=synergy_source()) if d["id"] == dataset_id)
        if item["status"] == "unavailable":
            raise HTTPException(503, {"dataset": dataset_id, "message": item["error"]})
        return item

    @router.get("/meta/data-dictionary")
    def dictionary():
        catalog = dataset_catalog(synergy=synergy_source())
        entries = dictionary_entries(catalog)
        return {"columns": entries, "undocumented_count": sum(not c["documented"] for c in entries)}

    return router
