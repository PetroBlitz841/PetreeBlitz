import { RegionCode as RegionCode } from "./utils/regions";

export interface Settings {
  region: RegionCode | "";
  planes: {
    traverse: boolean;
    radialLongitudinal: boolean;
    tangentialLongitudinal: boolean;
  };
}

export const DEFAULT_SETTINGS: Settings = {
  region: "",
  planes: {
    traverse: true,
    radialLongitudinal: false,
    tangentialLongitudinal: false,
  },
};

export interface Prediction {
  label: string;
  confidence: number;
}

export interface FeedbackPayload {
  sample_id: string;
  was_correct: boolean;
  correct_label?: string;
}

export interface Album {
  album_id: string;
  name: string;
  num_images: number;
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
