import { useState } from 'react';
import { Film } from 'lucide-react';

/** Thumbnail failure never hides the video: keep a stable lightweight placeholder. */
export function GalleryVideoTile({ thumbnail, filename }: { thumbnail: string; filename: string }) {
  const [failed, setFailed] = useState(false);
  if (failed || !thumbnail.startsWith('/api/gallery/video-thumb/')) {
    return <div className="gallery-media-placeholder"><Film size={26} /><small>Video</small></div>;
  }
  return (
    <img src={thumbnail} alt={`Video preview: ${filename}`}
      loading="lazy" decoding="async" onError={() => setFailed(true)} />
  );
}
