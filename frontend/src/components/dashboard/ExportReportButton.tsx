import { useRef, useState } from "react";
import {
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import PictureAsPdfIcon from "@mui/icons-material/PictureAsPdf";
import TableChartIcon from "@mui/icons-material/TableChart";

export default function ExportReportButton() {
  const [anchor, setAnchor] = useState<null | HTMLElement>(null);
  const [menuWidth, setMenuWidth] = useState<number | undefined>(undefined);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleOpen = (e: React.MouseEvent<HTMLButtonElement>) => {
    setMenuWidth(buttonRef.current?.offsetWidth);
    setAnchor(e.currentTarget);
  };

  return (
    <>
      <Button
        ref={buttonRef}
        variant="outlined"
        size="small"
        startIcon={<FileDownloadIcon />}
        endIcon={<KeyboardArrowDownIcon />}
        onClick={handleOpen}
      >
        Export Report
      </Button>
      <Menu
        anchorEl={anchor}
        open={Boolean(anchor)}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: { maxWidth: menuWidth },
          },
        }}
      >
        <MenuItem
          dense
          onClick={() => {
            setAnchor(null);
            window.open("/api/export/report?format=pdf", "_blank");
          }}
        >
          <ListItemIcon>
            <PictureAsPdfIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="PDF Report"
            slotProps={{ primary: { fontSize: "0.875rem" } }}
          />
        </MenuItem>
        <MenuItem
          dense
          onClick={() => {
            setAnchor(null);
            window.open("/api/export/report?format=csv", "_blank");
          }}
        >
          <ListItemIcon>
            <TableChartIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="CSV Spreadsheet"
            slotProps={{ primary: { fontSize: "0.875rem" } }}
          />
        </MenuItem>
      </Menu>
    </>
  );
}
