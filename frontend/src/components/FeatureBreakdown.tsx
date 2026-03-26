import React, { useState } from "react";
import {
  Box,
  Stack,
  Typography,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  LinearProgress,
  Divider,
} from "@mui/material";
import {
  ExpandMore,
  CheckCircle,
  Cancel,
  InfoOutlined,
} from "@mui/icons-material";
import { IAWAFeatureResult, FeatureCorrection } from "../types";

// ── category colours ──────────────────────────────────────────────────────────
const CATEGORY_COLOR: Record<string, string> = {
  "Growth Rings": "#795548",
  Porosity: "#1565C0",
  Vessels: "#2E7D32",
  "Vessel Pits": "#558B2F",
  Fibers: "#F57F17",
  Parenchyma: "#6A1B9A",
  Rays: "#00838F",
  Crystals: "#AD1457",
};

const IMPORTANCE_LABELS: Record<number, string> = {
  0.5: "Low",
  1.0: "Normal",
  1.5: "High",
  2.0: "Critical",
};

// ── helpers ───────────────────────────────────────────────────────────────────

function groupByCategory(
  features: IAWAFeatureResult[],
): Record<string, IAWAFeatureResult[]> {
  return features.reduce<Record<string, IAWAFeatureResult[]>>((acc, f) => {
    (acc[f.category] ??= []).push(f);
    return acc;
  }, {});
}

function confidenceColor(c: number): "success" | "warning" | "error" {
  return c >= 0.75 ? "success" : c >= 0.5 ? "warning" : "error";
}

// ── sub-components ────────────────────────────────────────────────────────────

interface FeatureRowProps {
  feature: IAWAFeatureResult;
  correction: FeatureCorrection | undefined;
  onCorrectionChange: (c: FeatureCorrection | null) => void;
}

function FeatureRow({
  feature,
  correction,
  onCorrectionChange,
}: FeatureRowProps) {
  const isMarkedIncorrect = correction !== undefined;
  const importance = correction?.importance_weight ?? 1.0;

  const handleIncorrectToggle = () => {
    if (isMarkedIncorrect) {
      // remove correction
      onCorrectionChange(null);
    } else {
      onCorrectionChange({
        feature_id: feature.id,
        is_present: !feature.is_present, // flip – expert says the opposite
        importance_weight: 1.0,
      });
    }
  };

  const handleImportanceChange = (
    _: React.MouseEvent<HTMLElement>,
    val: number | null,
  ) => {
    if (val === null) return;
    onCorrectionChange({
      feature_id: feature.id,
      is_present: correction?.is_present ?? !feature.is_present,
      importance_weight: val,
    });
  };

  return (
    <Box
      sx={{
        py: 1,
        px: 0.5,
        borderRadius: 1,
        bgcolor: isMarkedIncorrect ? "error.50" : "transparent",
        transition: "background-color 0.2s",
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap">
        {/* Presence indicator */}
        {feature.is_present ? (
          <CheckCircle fontSize="small" color="success" />
        ) : (
          <Cancel fontSize="small" color="disabled" />
        )}

        {/* Feature name + info tooltip */}
        <Typography variant="body2" sx={{ flex: 1, minWidth: 120 }}>
          {feature.name}
          <Tooltip title={feature.description} arrow placement="top">
            <InfoOutlined
              fontSize="inherit"
              sx={{
                ml: 0.5,
                verticalAlign: "middle",
                color: "text.disabled",
                cursor: "help",
              }}
            />
          </Tooltip>
        </Typography>

        {/* Confidence bar + chip */}
        <Stack
          direction="row"
          alignItems="center"
          spacing={0.75}
          sx={{ minWidth: 130 }}
        >
          <LinearProgress
            variant="determinate"
            value={feature.confidence * 100}
            color={confidenceColor(feature.confidence)}
            sx={{ flex: 1, height: 6, borderRadius: 3 }}
          />
          <Chip
            label={`${(feature.confidence * 100).toFixed(0)}%`}
            size="small"
            color={confidenceColor(feature.confidence)}
            variant="outlined"
            sx={{ minWidth: 50, height: 20, fontSize: "0.7rem" }}
          />
        </Stack>

        {/* "Incorrect" toggle */}
        <Tooltip
          title={
            isMarkedIncorrect
              ? "Remove correction"
              : "Mark as incorrectly identified"
          }
        >
          <Chip
            label={isMarkedIncorrect ? "Corrected" : "Mark incorrect"}
            size="small"
            color={isMarkedIncorrect ? "error" : "default"}
            variant={isMarkedIncorrect ? "filled" : "outlined"}
            onClick={handleIncorrectToggle}
            sx={{ cursor: "pointer", height: 22, fontSize: "0.7rem" }}
          />
        </Tooltip>

        {/* Importance selector – only visible when correction active */}
        {isMarkedIncorrect && (
          <ToggleButtonGroup
            size="small"
            value={importance}
            exclusive
            onChange={handleImportanceChange}
            sx={{ height: 22 }}
          >
            {[0.5, 1.0, 1.5, 2.0].map((w) => (
              <ToggleButton
                key={w}
                value={w}
                sx={{ px: 0.75, py: 0, fontSize: "0.65rem", lineHeight: 1 }}
              >
                {IMPORTANCE_LABELS[w]}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        )}
      </Stack>
    </Box>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

interface FeatureBreakdownProps {
  features: IAWAFeatureResult[];
  corrections: FeatureCorrection[];
  onCorrectionsChange: (corrections: FeatureCorrection[]) => void;
}

export default function FeatureBreakdown({
  features,
  corrections,
  onCorrectionsChange,
}: FeatureBreakdownProps) {
  const [expandedCategories, setExpandedCategories] = useState<
    Record<string, boolean>
  >(
    // expand all by default
    () => Object.fromEntries(Object.keys(CATEGORY_COLOR).map((k) => [k, true])),
  );

  const grouped = groupByCategory(features);
  const correctionMap = new Map(corrections.map((c) => [c.feature_id, c]));

  const handleFeatureCorrection =
    (featureId: number) => (c: FeatureCorrection | null) => {
      if (c === null) {
        onCorrectionsChange(
          corrections.filter((x) => x.feature_id !== featureId),
        );
      } else {
        onCorrectionsChange(
          corrections.filter((x) => x.feature_id !== featureId).concat(c),
        );
      }
    };

  const presentCount = features.filter((f) => f.is_present).length;
  const correctionCount = corrections.length;

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={1} mb={1}>
        <Typography variant="h6">IAWA Feature Analysis</Typography>
        <Chip
          label={`${presentCount} / ${features.length} detected`}
          size="small"
          color="primary"
          variant="outlined"
        />
        {correctionCount > 0 && (
          <Chip
            label={`${correctionCount} correction${correctionCount > 1 ? "s" : ""}`}
            size="small"
            color="warning"
          />
        )}
      </Stack>
      <Typography variant="body2" color="text.secondary" mb={1.5}>
        Features derived from IAWA standard wood anatomy assessment. Toggle any
        feature as incorrectly identified and adjust its importance weight to
        guide model learning.
      </Typography>

      {/* Per-category accordions */}
      {Object.entries(grouped).map(([category, catFeatures]) => {
        const color = CATEGORY_COLOR[category] ?? "#555";
        const catPresent = catFeatures.filter((f) => f.is_present).length;
        const isExpanded = expandedCategories[category] ?? true;

        return (
          <Accordion
            key={category}
            expanded={isExpanded}
            onChange={(_, v) =>
              setExpandedCategories((prev) => ({ ...prev, [category]: v }))
            }
            disableGutters
            elevation={0}
            sx={{
              border: "1px solid",
              borderColor: "divider",
              mb: 1,
              "&:before": { display: "none" },
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMore />}
              sx={{ minHeight: 40 }}
            >
              <Stack direction="row" alignItems="center" spacing={1}>
                <Box
                  sx={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    bgcolor: color,
                    flexShrink: 0,
                  }}
                />
                <Typography variant="subtitle2">{category}</Typography>
                <Chip
                  label={`${catPresent}/${catFeatures.length}`}
                  size="small"
                  sx={{
                    height: 18,
                    fontSize: "0.65rem",
                    bgcolor: color,
                    color: "#fff",
                  }}
                />
              </Stack>
            </AccordionSummary>
            <AccordionDetails sx={{ pt: 0 }}>
              {catFeatures.map((feat, idx) => (
                <React.Fragment key={feat.id}>
                  {idx > 0 && <Divider sx={{ my: 0.25 }} />}
                  <FeatureRow
                    feature={feat}
                    correction={correctionMap.get(feat.id)}
                    onCorrectionChange={handleFeatureCorrection(feat.id)}
                  />
                </React.Fragment>
              ))}
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );
}
