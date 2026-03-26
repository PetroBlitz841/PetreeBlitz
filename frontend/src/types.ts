import { RegionCode as RegionCode } from "./utils/regions";

export const PLANE_METADATA = {
  traverse: {
    label: "Transverse",
    description: "Cross-section cut perpendicular to the grain",
  },
  radialLongitudinal: {
    label: "Radial Longitudinal",
    description: "Cut parallel to the rays, along the radius",
  },
  tangentialLongitudinal: {
    label: "Tangential Longitudinal",
    description: "Cut tangent to the growth rings",
  },
} as const;

export type Plane = keyof typeof PLANE_METADATA;

export const PLANES: {
  [K in Plane]: (typeof PLANE_METADATA)[K] & { active: boolean };
} = Object.fromEntries(
  Object.entries(PLANE_METADATA).map(([key, value]) => [
    key,
    { ...value, active: key === "traverse" },
  ]),
) as { [K in Plane]: (typeof PLANE_METADATA)[K] & { active: boolean } };

export interface Settings {
  region: RegionCode | "";
  planes: Record<Plane, boolean>;
}

const defaultPlanes: Record<Plane, boolean> = Object.fromEntries(
  Object.keys(PLANES).map((key) => [key, PLANES[key as Plane].active]),
) as Record<Plane, boolean>;

export const DEFAULT_SETTINGS: Settings = {
  region: "",
  planes: defaultPlanes,
};

export interface Prediction {
  label: string;
  confidence: number;
}

export interface IAWAFeatureResult {
  id: number;
  name: string;
  category: string;
  description: string;
  is_present: boolean;
  confidence: number;
}

export interface FeatureSpeciesSupport {
  [species: string]: {
    matched: number[];
    mismatched: number[];
  };
}

export interface FeatureCorrection {
  feature_id: number;
  is_present: boolean;
  importance_weight: number; // 0.5 | 0.75 | 1.0 | 1.5 | 2.0
}

export interface FeedbackPayload {
  sample_id: string;
  was_correct: boolean;
  correct_label?: string;
  feature_corrections?: FeatureCorrection[];
}

export interface TaxonomyEntry {
  genus: string;
  family: string;
  order: string;
}

export interface Album {
  album_id: string;
  name: string;
  num_images: number;
  taxonomy?: TaxonomyEntry;
}

export interface Image {
  sample_id: string;
  image_url: string;
  predictions: Prediction[];
  feedback?: {
    was_correct: boolean;
    correct_label?: string;
  };
  timestamp: string;
}
