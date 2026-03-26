import { Card, CardContent, Typography } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

interface UploadCardProps {
  dragOver: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onClick: () => void;
  fileName?: string;
}

export default function UploadCard({
  dragOver,
  onDragOver,
  onDragLeave,
  onDrop,
  onClick,
  fileName,
}: UploadCardProps) {
  return (
    <Card
      sx={{
        border: dragOver ? "2px dashed #4caf50" : "2px dashed #ccc",
        backgroundColor: dragOver ? "green.50" : "white",
        cursor: "pointer",
        width: "100%",
      }}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
    >
      <CardContent sx={{ textAlign: "center", py: 4 }}>
        <CloudUploadIcon sx={{ fontSize: 48, color: "primary.main" }} />
        <Typography variant="h6" sx={{ mt: 1 }}>
          {fileName || "Drag & drop an image here or click to select"}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Supported formats: JPG, PNG, GIF
        </Typography>
      </CardContent>
    </Card>
  );
}
