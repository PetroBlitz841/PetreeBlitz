import { useState, useMemo } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Button,
  Box,
  TextField,
  Chip,
  Collapse,
  Divider,
  Tooltip,
} from "@mui/material";
import PublicIcon from "@mui/icons-material/Public";
import { Settings } from "../types";
import {
  L1,
  l2ByParent,
  l3ByParent,
  byCode,
  resolveAncestry,
  RegionCode,
} from "../utils/regions";
import TDWGMap from "./TDWGMap";

const L1_COLORS: Record<string, string> = {
  "1": "#4e8c3f",
  "2": "#c4813a",
  "3": "#4a7abf",
  "4": "#7b5fbf",
  "5": "#bf6040",
  "6": "#2fa8a4",
  "7": "#b03a3a",
  "8": "#b8a020",
  "9": "#7a9db0",
};

interface SettingsDialogProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (settings: Settings) => void;
  initialSettings: Settings;
}

export default function SettingsDialog({
  open,
  onCancel,
  onSubmit,
  initialSettings,
}: SettingsDialogProps) {
  const [settings, setSettings] = useState<Settings>(initialSettings);
  const [codeInput, setCodeInput] = useState("");
  const [codeError, setCodeError] = useState("");
  const [planesError, setPlanesError] = useState("");
  const [showMap, setShowMap] = useState(true);

  // Derived ancestry from the stored code
  const ancestry = useMemo(
    () => resolveAncestry(settings.region),
    [settings.region],
  );

  // Cascading dropdown options
  const l2Options = useMemo(
    () => (ancestry.l1 ? (l2ByParent[ancestry.l1.code] ?? []) : []),
    [ancestry.l1],
  );
  const l3Options = useMemo(
    () => (ancestry.l2 ? (l3ByParent[ancestry.l2.code] ?? []) : []),
    [ancestry.l2],
  );

  // Codes to soft-highlight on map (all L3 siblings within the selected L2)
  const siblingCodes = useMemo(() => {
    if (!ancestry.l2) return [];
    return (l3ByParent[ancestry.l2.code] ?? []).map(
      (r) => r.code,
    ) as RegionCode[];
  }, [ancestry.l2]);

  const setRegion = (code: Settings["region"]) =>
    setSettings((s: Settings) => ({ ...s, region: code }));

  const clearRegion = () => {
    setRegion("");
    setCodeInput("");
    setCodeError("");
  };

  const handleClose = () => {
    setSettings(initialSettings);
    setCodeInput("");
    setCodeError("");
    onCancel();
  };

  const handleSubmit = () => {
    if (settings.region && !(settings.region in byCode)) {
      setCodeError(`"${settings.region}" is not a valid TDWG code`);
      return;
    }

    const atLeastOnePlane = Object.values(settings.planes).some((v) => v);
    if (!atLeastOnePlane) {
      setPlanesError("At least one wood plane must be selected.");
      return;
    }

    onSubmit(settings);
  };

  const handleL1Change = (code: Settings["region"]) => {
    setRegion(code);
    setCodeInput(code);
    setCodeError("");
  };

  const handleL2Change = (code: Settings["region"]) => {
    setRegion(code);
    setCodeInput(code);
    setCodeError("");
  };

  const handleL3Change = (code: Settings["region"]) => {
    setRegion(code);
    setCodeInput(code);
    setCodeError("");
  };

  const handleCodeInput = (raw: string) => {
    setCodeInput(raw);
    const trimmed = raw.trim().toUpperCase();
    if (!trimmed) {
      setCodeError("");
      setRegion("");
      return;
    }
    const match = byCode[trimmed];
    if (match) {
      setCodeError("");
      setRegion(trimmed as Settings["region"]);
    } else {
      setCodeError(`"${trimmed}" is not a valid TDWG code`);
    }
  };

  const handleMapClick = (code: Settings["region"]) => {
    const r = byCode[code];
    if (!r) return;
    setRegion(code);
    setCodeInput(code);
    setCodeError("");
  };

  const handlePlaneChange = (key: keyof Settings["planes"]) => {
    setSettings((s) => {
      const newPlanes = { ...s.planes, [key]: !s.planes[key] };
      const atLeastOneSelected = Object.values(newPlanes).some((v) => v);
      setPlanesError(
        atLeastOneSelected ? "" : "At least one wood plane must be selected.",
      );
      return { ...s, planes: newPlanes };
    });
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Settings</DialogTitle>

      <DialogContent
        sx={{ pt: 2, display: "flex", flexDirection: "column", gap: 2.5 }}
      >
        {/* Region selection */}
        <Box>
          {/* Header row */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mb: 1,
            }}
          >
            <Typography variant="subtitle2">
              Region (TDWG World Geographical Scheme)
            </Typography>

            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              {/* Map toggle */}
              <Tooltip title={showMap ? "Hide map" : "Show map"}>
                <PublicIcon
                  fontSize="small"
                  sx={{
                    cursor: "pointer",
                    ml: 0.5,
                    color: showMap ? "primary.main" : "text.disabled",
                  }}
                  onClick={() => setShowMap((v) => !v)}
                />
              </Tooltip>
            </Box>
          </Box>

          {/* Map */}
          <Collapse in={showMap}>
            <Box sx={{ mb: 1.5 }}>
              <TDWGMap
                selectedCode={settings.region}
                highlightCodes={siblingCodes}
                onRegionClick={handleMapClick}
                height={220}
              />
            </Box>
          </Collapse>

          {/* L1 – Continent */}
          <FormControl fullWidth size="small" sx={{ mb: 1 }}>
            <InputLabel>Continent (Level 1)</InputLabel>
            <Select
              value={ancestry.l1?.code ?? ""}
              label="Continent (Level 1)"
              onChange={(e) =>
                handleL1Change(e.target.value as Settings["region"])
              }
            >
              <MenuItem value="">
                <em>Any continent</em>
              </MenuItem>
              {L1.map((r) => (
                <MenuItem key={r.code} value={r.code}>
                  <Box
                    component="span"
                    sx={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 1,
                    }}
                  >
                    <Box
                      component="span"
                      sx={{
                        width: 10,
                        height: 10,
                        borderRadius: "50%",
                        bgcolor: L1_COLORS[r.code],
                        flexShrink: 0,
                      }}
                    />
                    {r.code} · {r.name}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* L2 – Subregion */}
          <FormControl
            fullWidth
            size="small"
            sx={{ mb: 1 }}
            disabled={!ancestry.l1}
          >
            <InputLabel>Subregion (Level 2)</InputLabel>
            <Select
              value={ancestry.l2?.code ?? ""}
              label="Subregion (Level 2)"
              onChange={(e) =>
                handleL2Change(e.target.value as Settings["region"])
              }
            >
              <MenuItem value="">
                <em>Any subregion</em>
              </MenuItem>
              {l2Options.map((r) => (
                <MenuItem key={r.code} value={r.code}>
                  {r.code} · {r.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* L3 – Botanical Country */}
          <FormControl
            fullWidth
            size="small"
            sx={{ mb: 1 }}
            disabled={!ancestry.l2}
          >
            <InputLabel>Botanical Country (Level 3)</InputLabel>
            <Select
              value={ancestry.l3?.code ?? ""}
              label="Botanical Country (Level 3)"
              onChange={(e) =>
                handleL3Change(e.target.value as Settings["region"])
              }
            >
              <MenuItem value="">
                <em>Any country</em>
              </MenuItem>
              {l3Options.map((r) => (
                <MenuItem key={r.code} value={r.code}>
                  {r.code} · {r.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Divider sx={{ my: 1 }}>
            <Typography variant="caption" color="text.secondary">
              or enter code directly
            </Typography>
          </Divider>

          {/* Direct code input */}
          <TextField
            fullWidth
            size="small"
            label="TDWG code"
            placeholder="e.g. GER, 14, TUR, BZS…"
            value={codeInput}
            onChange={(e) => handleCodeInput(e.target.value)}
            error={!!codeError}
            helperText={
              codeError ||
              (settings.region && !codeError
                ? `${byCode[settings.region]?.name} · Level ${byCode[settings.region]?.level}`
                : "")
            }
            slotProps={{ input: { style: { fontFamily: "monospace" } } }}
          />

          {/* Breadcrumb chips */}
          {settings.region && (
            <Box
              sx={{
                mt: 1,
                display: "flex",
                alignItems: "center",
                gap: 0.75,
                flexWrap: "wrap",
              }}
            >
              {ancestry.l1 && (
                <Chip
                  size="small"
                  label={`${ancestry.l1.code} · ${ancestry.l1.name}`}
                  sx={{
                    bgcolor: L1_COLORS[ancestry.l1.code] + "22",
                    fontSize: 11,
                  }}
                />
              )}
              {ancestry.l2 && (
                <Chip
                  size="small"
                  variant="outlined"
                  label={`${ancestry.l2.code} · ${ancestry.l2.name}`}
                  sx={{ fontSize: 11 }}
                />
              )}
              {ancestry.l3 && (
                <Chip
                  size="small"
                  color="primary"
                  label={`${ancestry.l3.code} · ${ancestry.l3.name}`}
                  sx={{ fontSize: 11 }}
                />
              )}
              <Button
                size="small"
                onClick={clearRegion}
                sx={{ ml: "auto", minWidth: 0 }}
              >
                Clear
              </Button>
            </Box>
          )}
        </Box>

        {/* Wood Planes */}
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Wood Planes
          </Typography>
          <FormGroup>
            <FormControlLabel
              control={
                <Checkbox
                  checked={settings.planes.traverse}
                  onChange={() => handlePlaneChange("traverse")}
                />
              }
              label="Transverse Section (TS)"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={settings.planes.radialLongitudinal}
                  onChange={() => handlePlaneChange("radialLongitudinal")}
                />
              }
              label="Radial Longitudinal Section (RLS)"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={settings.planes.tangentialLongitudinal}
                  onChange={() => handlePlaneChange("tangentialLongitudinal")}
                />
              }
              label="Tangential Longitudinal Section (TLS)"
            />
          </FormGroup>
          {planesError && (
            <Typography variant="caption" color="error" sx={{ mt: 0.5 }}>
              {planesError}
            </Typography>
          )}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={!!codeError || !!planesError}
        >
          Save Settings
        </Button>
      </DialogActions>
    </Dialog>
  );
}
