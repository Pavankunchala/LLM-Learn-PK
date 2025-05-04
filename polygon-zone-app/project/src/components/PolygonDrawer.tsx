import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { X, Undo, CheckCircle, Plus } from 'lucide-react';
import { useDetectionStore } from '../store/detectionStore';
import { Polygon, Point } from '../types/detection';

interface PolygonDrawerProps {
  videoRef: React.RefObject<HTMLVideoElement>;
}

// Generate a color for each polygon
const getPolygonColor = (index: number) => {
  const colors = [
    'rgba(59, 130, 246, 0.5)',  // blue
    'rgba(249, 115, 22, 0.5)',  // orange
    'rgba(16, 185, 129, 0.5)',  // green
    'rgba(236, 72, 153, 0.5)',  // pink
    'rgba(168, 85, 247, 0.5)',  // purple
    'rgba(234, 179, 8, 0.5)',   // yellow
  ];
  return colors[index % colors.length];
};

const PolygonDrawer: React.FC<PolygonDrawerProps> = ({ videoRef }) => {
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPolygon, setCurrentPolygon] = useState<Point[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const { 
    polygons, 
    addPolygon, 
    clearPolygons,
    removeLastPolygon 
  } = useDetectionStore();

  // Clear the canvas and redraw all polygons
  const redrawPolygons = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions to match video
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw completed polygons
    polygons.forEach((polygon, index) => {
      if (polygon.points.length < 3) return;
      
      ctx.fillStyle = getPolygonColor(index);
      ctx.strokeStyle = getPolygonColor(index).replace('0.5', '0.8');
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      ctx.moveTo(
        polygon.points[0].x * canvas.width,
        polygon.points[0].y * canvas.height
      );
      
      for (let i = 1; i < polygon.points.length; i++) {
        ctx.lineTo(
          polygon.points[i].x * canvas.width,
          polygon.points[i].y * canvas.height
        );
      }
      
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      
      // Draw points
      polygon.points.forEach(point => {
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(
          point.x * canvas.width,
          point.y * canvas.height,
          4, 0, Math.PI * 2
        );
        ctx.fill();
      });
      
      // Add zone label
      const centerX = polygon.points.reduce((sum, p) => sum + p.x, 0) / polygon.points.length * canvas.width;
      const centerY = polygon.points.reduce((sum, p) => sum + p.y, 0) / polygon.points.length * canvas.height;
      
      ctx.fillStyle = 'white';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Zone ${index + 1}`, centerX, centerY);
    });
    
    // Draw current polygon being created
    if (currentPolygon.length > 0) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      ctx.moveTo(
        currentPolygon[0].x * canvas.width,
        currentPolygon[0].y * canvas.height
      );
      
      for (let i = 1; i < currentPolygon.length; i++) {
        ctx.lineTo(
          currentPolygon[i].x * canvas.width,
          currentPolygon[i].y * canvas.height
        );
      }
      
      // If we're actively drawing, add line to mouse position
      if (isDrawing && currentPolygon.length > 0) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = (lastMousePos.x - rect.left) / canvas.width;
        const mouseY = (lastMousePos.y - rect.top) / canvas.height;
        
        ctx.lineTo(mouseX * canvas.width, mouseY * canvas.height);
      }
      
      ctx.stroke();
      
      // Draw points for current polygon
      currentPolygon.forEach(point => {
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(
          point.x * canvas.width,
          point.y * canvas.height,
          4, 0, Math.PI * 2
        );
        ctx.fill();
      });
    }
  };

  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  
  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / canvas.width;
    const y = (e.clientY - rect.top) / canvas.height;
    
    // Add point to current polygon
    setCurrentPolygon([...currentPolygon, { x, y }]);
  };
  
  // Handle mouse move for drawing preview line
  const handleMouseMove = (e: React.MouseEvent) => {
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };
  
  // Complete the current polygon
  const completePolygon = () => {
    if (currentPolygon.length >= 3) {
      const newPolygon: Polygon = {
        id: Date.now().toString(),
        points: currentPolygon,
      };
      
      addPolygon(newPolygon);
      setCurrentPolygon([]);
      setIsDrawing(false);
    }
  };
  
  // Undo last point in current polygon
  const undoLastPoint = () => {
    if (currentPolygon.length > 0) {
      setCurrentPolygon(currentPolygon.slice(0, -1));
    }
  };
  
  // Start a new polygon
  const startNewPolygon = () => {
    setCurrentPolygon([]);
    setIsDrawing(true);
  };
  
  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      redrawPolygons();
    };
    
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  // Update canvas when polygons or drawing state changes
  useEffect(() => {
    redrawPolygons();
    
    // Set up animation frame for drawing the preview line
    let animationId: number;
    
    if (isDrawing) {
      const animate = () => {
        redrawPolygons();
        animationId = requestAnimationFrame(animate);
      };
      
      animationId = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [polygons, currentPolygon, isDrawing, lastMousePos]);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full cursor-crosshair"
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
      />
      
      <div className="absolute top-4 right-4 flex flex-col space-y-2">
        {!isDrawing ? (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="bg-primary-600 hover:bg-primary-500 text-white p-2 rounded-lg shadow-lg flex items-center"
            onClick={startNewPolygon}
          >
            <Plus size={16} className="mr-1" />
            <span className="text-sm">New Zone</span>
          </motion.button>
        ) : (
          <>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-success-600 hover:bg-success-500 text-white p-2 rounded-lg shadow-lg flex items-center"
              onClick={completePolygon}
              disabled={currentPolygon.length < 3}
            >
              <CheckCircle size={16} className="mr-1" />
              <span className="text-sm">Complete</span>
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gray-700 hover:bg-gray-600 text-white p-2 rounded-lg shadow-lg flex items-center"
              onClick={undoLastPoint}
              disabled={currentPolygon.length === 0}
            >
              <Undo size={16} className="mr-1" />
              <span className="text-sm">Undo Point</span>
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gray-700 hover:bg-gray-600 text-white p-2 rounded-lg shadow-lg flex items-center"
              onClick={() => setIsDrawing(false)}
            >
              <X size={16} className="mr-1" />
              <span className="text-sm">Cancel</span>
            </motion.button>
          </>
        )}
        
        {polygons.length > 0 && !isDrawing && (
          <>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-error-600 hover:bg-error-500 text-white p-2 rounded-lg shadow-lg flex items-center"
              onClick={clearPolygons}
            >
              <X size={16} className="mr-1" />
              <span className="text-sm">Clear All</span>
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gray-700 hover:bg-gray-600 text-white p-2 rounded-lg shadow-lg flex items-center"
              onClick={removeLastPolygon}
            >
              <Undo size={16} className="mr-1" />
              <span className="text-sm">Remove Last</span>
            </motion.button>
          </>
        )}
      </div>
      
      {isDrawing && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-gray-900 bg-opacity-75 p-2 rounded-lg">
          <p className="text-sm text-white">
            Click to add points. Complete with at least 3 points.
          </p>
        </div>
      )}
    </>
  );
};

export default PolygonDrawer;