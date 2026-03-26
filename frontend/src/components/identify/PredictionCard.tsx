import { useState } from "react";
import {
  Card,
  CardContent,
  Typography,
  Chip,
  Stack,
  Button,
  Box,
  LinearProgress,
  Collapse,
  IconButton,
  Tooltip,
  Divider,
} from "@mui/material";
import {
  CheckCircle,
  Cancel,
  ExpandMore,
  ExpandLess,
} from "@mui/icons-material";
import { Prediction, IAWAFeatureResult, FeatureSpeciesSupport } from "../../types";

interface PredictionCardProps {
  prediction: Prediction;
  index: number;
  feedbackLoading?: boolean;
  featureSupport?: FeatureSpeciesSupport;
  allFeatures?: IAWAFeatureResult[];
  onCorrect: (label: string) => void;
}

function matchColor(ratio: number): "success" | "warning" | "error" {
  return ratio >= 0.7 ? "success" : ratio >= 0.45 ? "warning" : "error";
}

export default function PredictionCard({
  prediction,
  index,
  feedbackLoading = false,
  featureSupport,
  allFeatures = [],
  onCorrect,
}: PredictionCardProps) {
  const [expanded, setExpanded] = useState(false);

  const confidenceColor: "success" | "warning" | "error" =
    prediction.confidence > 0.8
      ? "success"
      : prediction.confidence > 0.5
        ? "warning"
        : "error";

  const support = featureSupport?.[prediction.label];
  const featureMap = new Map(allFeatures.map((f) => [f.id, f]));
  const matched = (support?.matched ?? [])
    .map((id) => featureMap.get(id))
    .filter(Boolean) as IAWAFeatureResult[];
  const mismatched = (support?.mismatched ?? [])
    .map((id) => featureMap.get(id))
    .filter(Boolean) as IAWAFeatureResult[];
  const total = matched.length + mismatched.length;
  const ratio = total > 0 ? matched.length / total : 0;
  const hasFeatures = total > 0;

  return (
    <Card variant="outlined" sx={{ height: "100%" }}>
      <CardContent>
        <Stack direction="column" spacing={1.5}>
          {/* Header: rank + label + confidence */}
          <Stack direction="row" alignItems="flex-start" spacing={1}>
            <Chip
              label={`#${index + 1}`}
              size="small"
              variant="outlined"
              sx={{ fontWeight: 700, minWidth: 36 }}
            />
            <Box flex={1}>
              <Typography variant="subtitle1" fontWeight={700} lineHeight={1.3}>
                {prediction.label.replace(/_/g, " ")}
              </Typography>
              <Chip
                label={`${(prediction.confidence * 100).toFixed(1)}% confidence`}
                color={confidenceColor}
                size="small"
                sx={{ mt: 0.5, fontSize: "0.7rem" }}
              />
            </Box>
          </Stack>

          {/* Feature match bar */}
          {hasFeatures && (
            <Box>
              <Stack direction="row" alignItems="center" spacing={1}>
                <LinearProgress
                  variant="determinate"
                  value={ratio * 100}
                  color={matchColor(ratio)}
                  sx={{ flex: 1, height: 6, borderRadius: 3 }}
                />
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ minWidth: 72, textAlign: "right" }}
                >
                  {matched.length}/{total} features
                </Typography>
                <Tooltip title={expanded ? "Hide features" : "Show features"}>
                  <IconButton
                    size="small"
                    onClick={() => setExpanded((v) => !v)}
                    sx={{ p: 0.25 }}
                  >
                    {expanded ? (
                      <ExpandLess fontSize="small" />
                    ) : (
                      <ExpandMore fontSize="small" />
                    )}
                  </IconButton>
                </Tooltip>
              </Stack>

              {/* Expandable feature detail */}
              <Collapse in={expanded}>
                <Box mt={1.25}>
                  {matched.length > 0 && (
                    <Box mb={0.75}>
                      <Typography
                        variant="caption"
                        color="success.main"
                        fontWeight={600}
                        display="block"
                        mb={0.5}
                      >
                        Matched ({matched.length})
                      </Typography>
                      <Stack spacing={0.25}>
                        {matched.map((f) => (
                          <Stack
                            key={f.id}
                            direction="row"
                            alignItems="center"
                            spacing={0.75}
                          >
                            <CheckCircle
                              color="success"
                              sx={{ fontSize: 12 }}
                            />
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {f.name}
                            </Typography>
                          </Stack>
                        ))}
                      </Stack>
                    </Box>
                  )}
                  {mismatched.length > 0 && (
                    <Box>
                      {matched.length > 0 && <Divider sx={{ mb: 0.75 }} />}
                      <Typography
                        variant="caption"
                        color="error.main"
                        fontWeight={600}
                        display="block"
                        mb={0.5}
                      >
                        Mismatched ({mismatched.length})
                      </Typography>
                      <Stack spacing={0.25}>
                        {mismatched.map((f) => (
                          <Stack
                            key={f.id}
                            direction="row"
                            alignItems="center"
                            spacing={0.75}
                          >
                            <Cancel color="error" sx={{ fontSize: 12 }} />
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {f.name}
                            </Typography>
                          </Stack>
                        ))}
                      </Stack>
                    </Box>
                  )}
                </Box>
              </Collapse>
            </Box>
          )}

          {/* Action */}
          <Button
            variant="outlined"
            color="success"
            size="small"
            startIcon={<CheckCircle />}
            onClick={() => onCorrect(prediction.label)}
            disabled={feedbackLoading}
            fullWidth
          >
            This is correct
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
}
