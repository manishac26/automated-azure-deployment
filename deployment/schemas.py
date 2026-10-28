"""Pydantic schemas for the intrusion detection API."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Dict[str, Union[float, int, str]] = Field(
        ..., description="Mapping from feature name to numeric value (all required)."
    )

    @field_validator("features")
    @classmethod
    def validate_floats(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("features must be a dictionary")

        for k, val in v.items():
            if val is None:
                raise ValueError(f"Invalid numeric value for feature '{k}': None")
            # Check for string representations of non-finite numbers
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
                    raise ValueError(f"Invalid numeric value for feature '{k}': {val}")
                # try to parse
                try:
                    fval = float(val)
                except Exception as exc:
                    raise ValueError(f"Invalid numeric value for feature '{k}'") from exc
                if math.isnan(fval) or math.isinf(fval):
                    raise ValueError(f"Invalid numeric value for feature '{k}': {val}")
                # store as float
                v[k] = float(fval)
                continue
            # numbers
            try:
                fval = float(val)
            except Exception as exc:
                raise ValueError(f"Invalid numeric value for feature '{k}'") from exc
            if math.isnan(fval) or math.isinf(fval):
                raise ValueError(f"Invalid numeric value for feature '{k}': {val}")
            if isinstance(val, bool):  # bool is subclass of int in Python
                raise ValueError(f"Invalid numeric value for feature '{k}'")
            v[k] = float(fval)
        return v


class PredictionResponse(BaseModel):
    predicted_label: int = Field(..., description="0 = BENIGN, 1 = ATTACK")
    predicted_label_name: str
    probability: Optional[float] = Field(
        None, description="Attack probability if available"
    )
    model_name: str
    feature_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class MetadataResponse(BaseModel):
    model_name: str
    model_path: str
    feature_names: List[str]
    feature_count: int
    label_map: Dict[str, int]
    preprocessing: str
    notes: Optional[str] = None
