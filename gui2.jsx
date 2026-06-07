import React, { useState, useRef, useEffect } from 'react';
import { Download, Upload, Link2, Trash2, RefreshCw, ZoomIn, ZoomOut, Grid, Eye, FileText, Image } from 'lucide-react';

const AirfoilAnalyzer = () => {
  const [airfoils, setAirfoils] = useState([]);
  const [selectedAirfoil, setSelectedAirfoil] = useState(null);
  const [urlInput, setUrlInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [zoom, setZoom] = useState(1);
  const [showGrid, setShowGrid] = useState(true);
  const [showCoords, setShowCoords] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [selectedForCompare, setSelectedForCompare] = useState([]);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const generateNACA4Digit = (m, p, t, numPoints = 100) => {
    const coords = [];
    for (let i = 0; i <= numPoints; i++) {
      const x = (1 - Math.cos((i / numPoints) * Math.PI)) / 2;
      const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);
      let yc, dyc_dx;
      if (x < p) {
        yc = m * (2 * p * x - x * x) / (p * p);
        dyc_dx = 2 * m * (p - x) / (p * p);
      } else {
        yc = m * ((1 - 2 * p) + 2 * p * x - x * x) / ((1 - p) * (1 - p));
        dyc_dx = 2 * m * (p - x) / ((1 - p) * (1 - p));
      }
      const theta = Math.atan(dyc_dx);
      coords.push([x - yt * Math.sin(theta), yc + yt * Math.cos(theta)]);
    }
    for (let i = numPoints; i >= 0; i--) {
      const x = (1 - Math.cos((i / numPoints) * Math.PI)) / 2;
      const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);
      let yc, dyc_dx;
      if (x < p) {
        yc = m * (2 * p * x - x * x) / (p * p);
        dyc_dx = 2 * m * (p - x) / (p * p);
      } else {
        yc = m * ((1 - 2 * p) + 2 * p * x - x * x) / ((1 - p) * (1 - p));
        dyc_dx = 2 * m * (p - x) / ((1 - p) * (1 - p));
      }
      const theta = Math.atan(dyc_dx);
      coords.push([x + yt * Math.sin(theta), yc - yt * Math.cos(theta)]);
    }
    return coords;
  };

  const loadSampleAirfoil = (name) => {
    let coords;
    if (name === 'NACA 0012') coords = generateNACA4Digit(0, 0, 0.12);
    else if (name === 'NACA 2412') coords = generateNACA4Digit(0.02, 0.4, 0.12);
    else if (name === 'NACA 4415') coords = generateNACA4Digit(0.04, 0.4, 0.15);
    else coords = generateNACA4Digit(0.035, 0.3, 0.118);
    
    setAirfoils(prev => [...prev, {
      id: Date.now(),
      name: name,
      coords: coords,
      source: 'sample'
    }]);
    setSelectedAirfoil({ id: Date.now(), name, coords, source: 'sample' });
  };

  const parseDatContent = (content, filename) => {
    const lines = content.trim().split('\n');
    const name = lines[0].trim() || filename.replace('.dat', '');
    const coords = lines.slice(1).map(line => {
      const parts = line.trim().split(/\s+/);
      return [parseFloat(parts[0]), parseFloat(parts[1])];
    }).filter(p => !isNaN(p[0]) && !isNaN(p[1]));
    return { name, coords };
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const { name, coords } = parseDatContent(event.target.result, file.name);
        const newAirfoil = { id: Date.now(), name, coords, source: 'file' };
        setAirfoils(prev => [...prev, newAirfoil]);
        setSelectedAirfoil(newAirfoil);
        setError('');
      } catch (err) {
        setError('Invalid DAT file format');
      }
    };
    reader.readAsText(file);
  };

  const handleUrlFetch = async () => {
    if (!urlInput.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(urlInput);
      const content = await response.text();
      const { name, coords } = parseDatContent(content, 'airfoil.dat');
      const newAirfoil = { id: Date.now(), name, coords, source: 'url', url: urlInput };
      setAirfoils(prev => [...prev, newAirfoil]);
      setSelectedAirfoil(newAirfoil);
      setUrlInput('');
      setError('');
    } catch (err) {
      setError('Failed to fetch airfoil data');
    }
    setLoading(false);
  };

  const calculateProperties = (coords) => {
    if (!coords) return null;
    const xCoords = coords.map(p => p[0]);
    const yCoords = coords.map(p => p[1]);
    const chord = Math.max(...xCoords) - Math.min(...xCoords);
    const thickness = Math.max(...yCoords) - Math.min(...yCoords);
    return {
      chord: chord.toFixed(4),
      thickness: thickness.toFixed(4),
      numPoints: coords.length
    };
  };

  const drawAirfoil = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, w, h);
    
    const m = 60;
    const pw = w - 2 * m;
    const ph = h - 2 * m;
    
    if (showGrid) {
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        ctx.beginPath();
        ctx.moveTo(m + i * pw / 10, m);
        ctx.lineTo(m + i * pw / 10, h - m);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(m, m + i * ph / 10);
        ctx.lineTo(w - m, m + i * ph / 10);
        ctx.stroke();
      }
    }
    
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(m, h / 2);
    ctx.lineTo(w - m, h / 2);
    ctx.moveTo(m, m);
    ctx.lineTo(m, h - m);
    ctx.stroke();
    
    const airfoilsToDraw = compareMode && selectedForCompare.length > 0 ? selectedForCompare : selectedAirfoil ? [selectedAirfoil] : [];
    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
    
    airfoilsToDraw.forEach((airfoil, idx) => {
      if (!airfoil?.coords) return;
      const coords = airfoil.coords;
      const xs = coords.map(p => p[0]);
      const ys = coords.map(p => p[1]);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const xRange = xMax - xMin;
      const yRange = Math.max(Math.abs(Math.min(...ys)), Math.abs(Math.max(...ys))) * 2.2 * zoom;
      const color = colors[idx % colors.length];
      
      ctx.strokeStyle = color;
      ctx.fillStyle = color + '30';
      ctx.lineWidth = 2;
      ctx.beginPath();
      coords.forEach((p, i) => {
        const x = m + ((p[0] - xMin) / xRange) * pw;
        const y = h / 2 - (p[1] / yRange) * ph;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fill();
      
      if (compareMode) {
        ctx.fillStyle = color;
        ctx.fillRect(w - m - 100, m + idx * 25, 20, 10);
        ctx.fillStyle = '#374151';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(airfoil.name, w - m - 75, m + idx * 25 + 9);
      }
    });
    
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(compareMode ? 'Airfoil Comparison' : selectedAirfoil?.name || '', w / 2, 30);
  };

  useEffect(() => {
    drawAirfoil();
  }, [selectedAirfoil, selectedForCompare, compareMode, zoom, showGrid, showCoords]);

  const downloadCSV = (airfoil) => {
    if (!airfoil) return;
    const csv = airfoil.coords.map(p => `${p[0]},${p[1]},0`).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${airfoil.name.replace(/\s+/g, '_')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadDAT = (airfoil) => {
    if (!airfoil) return;
    const dat = `${airfoil.name}\n` + airfoil.coords.map(p => `  ${p[0].toFixed(6)}  ${p[1].toFixed(6)}`).join('\n');
    const blob = new Blob([dat], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${airfoil.name.replace(/\s+/g, '_')}.dat`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadImage = () => {
    canvasRef.current?.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `airfoil_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  const toggleCompare = (airfoil) => {
    setSelectedForCompare(prev => {
      const exists = prev.find(a => a.id === airfoil.id);
      return exists ? prev.filter(a => a.id !== airfoil.id) : prev.length < 5 ? [...prev, airfoil] : prev;
    });
  };

  const properties = selectedAirfoil ? calculateProperties(selectedAirfoil.coords) : null;

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Airfoil Data Analyzer</h1>
              <p className="text-gray-600">Import, visualize, and analyze airfoil geometries</p>
            </div>
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${compareMode ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
            >
              {compareMode ? 'Exit Compare' : 'Compare Mode'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">Load from URL</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://example.com/airfoil.dat"
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  onKeyPress={(e) => e.key === 'Enter' && handleUrlFetch()}
                />
                <button
                  onClick={handleUrlFetch}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center gap-2"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Link2 className="w-4 h-4" />}
                  Fetch
                </button>
              </div>
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">Upload DAT File</label>
              <input ref={fileInputRef} type="file" accept=".dat,.txt" onChange={handleFileUpload} className="hidden" />
              <button onClick={() => fileInputRef.current?.click()} className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center justify-center gap-2">
                <Upload className="w-4 h-4" />
                Choose File
              </button>
            </div>
          </div>

          {error && <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{error}</div>}

          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Sample Airfoils</label>
            <div className="flex flex-wrap gap-2">
              {['NACA 0012', 'NACA 2412', 'NACA 4415', 'Clark Y'].map((name) => (
                <button key={name} onClick={() => loadSampleAirfoil(name)} className="px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 text-sm font-medium">
                  {name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 bg-white rounded-xl shadow-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">Library</h2>
              <span className="text-sm text-gray-500">{airfoils.length} loaded</span>
            </div>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {airfoils.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-8">No airfoils loaded yet</p>
              ) : (
                airfoils.map((airfoil) => (
                  <div
                    key={airfoil.id}
                    className={`p-3 rounded-lg border-2 cursor-pointer ${
                      selectedAirfoil?.id === airfoil.id ? 'border-blue-500 bg-blue-50' :
                      selectedForCompare.find(a => a.id === airfoil.id) ? 'border-green-500 bg-green-50' :
                      'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => !compareMode && setSelectedAirfoil(airfoil)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="font-medium text-sm text-gray-800">{airfoil.name}</h3>
                        <p className="text-xs text-gray-500 mt-1">{airfoil.coords.length} points</p>
                      </div>
                      <div className="flex gap-1">
                        {compareMode && (
                          <button
                            onClick={(e) => { e.stopPropagation(); toggleCompare(airfoil); }}
                            className={`p-1 rounded ${selectedForCompare.find(a => a.id === airfoil.id) ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-600'}`}
                          >
                            <Eye className="w-3 h-3" />
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setAirfoils(prev => prev.filter(a => a.id !== airfoil.id));
                            if (selectedAirfoil?.id === airfoil.id) setSelectedAirfoil(null);
                            setSelectedForCompare(prev => prev.filter(a => a.id !== airfoil.id));
                          }}
                          className="p-1 rounded bg-red-100 text-red-600 hover:bg-red-200"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="lg:col-span-3 space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-800">Visualization</h2>
                <div className="flex gap-2">
                  <button onClick={() => setShowGrid(!showGrid)} className={`p-2 rounded-lg ${showGrid ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}`}>
                    <Grid className="w-4 h-4" />
                  </button>
                  <button onClick={() => setShowCoords(!showCoords)} className={`p-2 rounded-lg ${showCoords ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}`}>
                    <FileText className="w-4 h-4" />
                  </button>
                  <button onClick={() => setZoom(Math.max(0.5, zoom - 0.1))} className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200">
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  <button onClick={() => setZoom(Math.min(2, zoom + 0.1))} className="p-2 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200">
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <button onClick={downloadImage} className="p-2 rounded-lg bg-purple-100 text-purple-600 hover:bg-purple-200">
                    <Image className="w-4 h-4" />
                  </button>
                </div>
              </div>
              
              <canvas ref={canvasRef} width={900} height={500} className="w-full border border-gray-200 rounded-lg" />
            </div>

            {properties && !compareMode && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-800">Properties</h2>
                  <div className="flex gap-2">
                    <button onClick={() => downloadCSV(selectedAirfoil)} className="px-3 py-1.5 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 text-sm font-medium flex items-center gap-1">
                      <Download className="w-3 h-3" />
                      CSV
                    </button>
                    <button onClick={() => downloadDAT(selectedAirfoil)} className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 text-sm font-medium flex items-center gap-1">
                      <Download className="w-3 h-3" />
                      DAT
                    </button>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-600 mb-1">Chord Length</p>
                    <p className="text-lg font-semibold text-gray-800">{properties.chord}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-600 mb-1">Thickness</p>
                    <p className="text-lg font-semibold text-gray-800">{properties.thickness}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-600 mb-1">Data Points</p>
                    <p className="text-lg font-semibold text-gray-800">{properties.numPoints}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AirfoilAnalyzer;
