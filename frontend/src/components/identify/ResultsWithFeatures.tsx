import { Box, Divider, Stack } from "@mui/material";
import FeatureBreakdown from "./FeatureBreakdown";
import FeatureSpeciesComparison from "./FeatureSpeciesComparison";
import {
  Prediction,
  IAWAFeatureResult,
  FeatureSpeciesSupport,
  FeatureCorrection,
} from "../../types";

interface ResultsWithFeaturesProps {
  results: Prediction[];
  features: IAWAFeatureResult[];
  featureSupport: FeatureSpeciesSupport;
  corrections: FeatureCorrection[];
  feedbackLoading?: boolean;
  onCorrectionsChange: (corrections: FeatureCorrection[]) => void;
  onCorrect: (label: string) => void;
  onNewSpecies: () => void;
}

export default function ResultsWithFeatures({
  results,
  features,
  featureSupport,
  corrections,
  feedbackLoading = false,
  onCorrectionsChange,
  onCorrect,
  onNewSpecies,
}: ResultsWithFeaturesProps) {
  if (results.length === 0 && features.length === 0) return null;

  return (
    <Stack direction="column" spacing={3} sx={{ width: "100%" }}>
      {/* 1. Feature breakdown with inline correction controls */}
      {features.length > 0 && (
        <Box>
          <FeatureBreakdown
            features={features}
            corrections={corrections}
            onCorrectionsChange={onCorrectionsChange}
          />
        </Box>
      )}

      {/* 2. Identification results with feature alignment + action buttons */}
      {results.length > 0 && (
        <>
          {features.length > 0 && <Divider />}
          <FeatureSpeciesComparison
            predictions={results}
            featureSupport={featureSupport}
            allFeatures={features}
            feedbackLoading={feedbackLoading}
            onCorrect={onCorrect}
            onNewSpecies={onNewSpecies}
          />
        </>
      )}
    </Stack>
  );
}
