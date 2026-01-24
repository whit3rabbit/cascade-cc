// Elements
const container = document.getElementById('graph-container');
const infoElement = document.getElementById('info');
const searchInput = document.getElementById('search-input');
const fileInput = document.getElementById('file-input');
const loadFileBtn = document.getElementById('load-file-btn');
const filterSlider = document.getElementById('centrality-filter');
const filterValue = document.getElementById('filter-value');
const resetBtn = document.getElementById('reset-btn');
const layoutBtns = document.querySelectorAll('.layout-btn');

let graph;
let renderer;
let allNodesData = [];

// Initialize
async function init() {
    try {
        const urlParams = new URLSearchParams(window.location.search);
        const fileName = urlParams.get('file') || 'graph_map.json';
        const metadataBase = '../cascade_graph_analysis/metadata/';

        let rawData;
        if (window.GRAPH_DATA && !urlParams.has('file')) {
            console.log('[*] Using embedded graph data');
            rawData = window.GRAPH_DATA;
        } else {
            console.log(`[*] Fetching graph data from ${fileName}...`);
            const response = await fetch(metadataBase + fileName);
            if (!response.ok) throw new Error(`Failed to load ${fileName}: ${response.statusText}`);
            rawData = await response.json();
        }

        allNodesData = rawData;

        // Initialize Graphology graph
        graph = new graphology.Graph();

        // Process data
        updateGraphFromFilter();

        // Render - Note the capitalized Sigma
        renderer = new Sigma(graph, container, {
            renderEdgeLabels: false,
            labelSize: 10,
            labelWeight: 'bold',
            defaultNodeColor: '#58a6ff',
            defaultEdgeColor: '#30363d'
        });

        // Event Listeners
        renderer.on('enterNode', () => {
            container.style.cursor = 'pointer';
        });
        renderer.on('leaveNode', () => {
            container.style.cursor = 'default';
        });

        renderer.on('clickNode', ({ node }) => {
            const data = graph.getNodeAttributes(node);
            showNodeInfo(data);
        });

        renderer.on('clickStage', () => {
            resetInfo();
        });

        filterSlider.addEventListener('input', (e) => {
            filterValue.innerText = `Top ${e.target.value}%`;
        });

        filterSlider.addEventListener('change', () => {
            updateGraphFromFilter();
        });

        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            if (!term) {
                renderer.setSetting('nodeReducer', null);
                renderer.setSetting('edgeReducer', null);
            } else {
                const matched = new Set();
                graph.forEachNode((nodeId, attrs) => {
                    if ((attrs.label || '').toLowerCase().includes(term)) matched.add(nodeId);
                });

                const neighbors = new Set();
                const focusedEdges = new Set();
                matched.forEach(nodeId => {
                    graph.forEachNeighbor(nodeId, neighborId => {
                        neighbors.add(neighborId);
                    });
                    graph.forEachEdge(nodeId, edgeId => {
                        focusedEdges.add(edgeId);
                    });
                });

                const focusNodes = new Set([...matched, ...neighbors]);

                renderer.setSetting('nodeReducer', (node, data) => {
                    const res = { ...data };
                    if (!focusNodes.has(node)) {
                        res.label = '';
                        res.color = '#333';
                        res.zIndex = 0;
                    } else if (matched.has(node)) {
                        res.color = '#ffb86b';
                        res.zIndex = 2;
                    } else {
                        res.color = '#666';
                        res.zIndex = 1;
                    }
                    return res;
                });

                renderer.setSetting('edgeReducer', (edge, data) => {
                    const res = { ...data };
                    if (!focusedEdges.has(edge)) {
                        res.hidden = true;
                        res.color = '#1f1f1f';
                    } else {
                        res.hidden = false;
                        res.color = '#ffb86b';
                        res.size = 1.5;
                    }
                    return res;
                });
            }
        });

        layoutBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                layoutBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                if (btn.dataset.layout === 'circular') {
                    if (window.FA2Layout) window.FA2Layout.stop();
                    graphologyLibrary.layout.circular.assign(graph);
                } else if (btn.dataset.layout === 'force') {
                    if (window.FA2Layout) window.FA2Layout.stop();
                    window.FA2Layout = new graphologyLibrary.layoutForceAtlas2.FA2Layout(graph, {
                        settings: { gravity: 1 }
                    });
                    window.FA2Layout.start();
                }
            });
        });

        resetBtn.addEventListener('click', () => {
            renderer.getCamera().animatedReset();
            resetInfo();
        });

        loadFileBtn.addEventListener('click', () => {
            const fileName = fileInput.value || 'graph_map.json';
            const url = new URL(window.location.href);
            url.searchParams.set('file', fileName);
            window.location.href = url.href;
        });

        // Default layout: Circular for very large graphs, ForceAtlas2 otherwise
        if (graph.order > 500) {
            console.log(`[*] Large graph detected (${graph.order} nodes), using Circular layout.`);
            graphologyLibrary.layout.circular.assign(graph);
        } else {
            // Use Web Worker for ForceAtlas2 layout to keep UI responsive
            if (window.FA2Layout) window.FA2Layout.stop();
            window.FA2Layout = new graphologyLibrary.layoutForceAtlas2.FA2Layout(graph, {
                settings: { gravity: 1 }
            });
            window.FA2Layout.start();
            setTimeout(() => window.FA2Layout.stop(), 5000); // Auto-stop after 5s or when user clicks
        }

    } catch (err) {
        console.error('Initialization error:', err);
        infoElement.innerHTML = `<h1>Error</h1><p>${err.message}</p><p>Check console for details.</p>`;
    }
}

function updateGraphFromFilter() {
    const percent = parseInt(filterSlider.value) / 100;
    const sorted = [...allNodesData].sort((a, b) => b.centrality - a.centrality);
    const cutOff = Math.max(10, Math.floor(sorted.length * percent));
    const activeData = sorted.slice(0, cutOff);
    const activeIds = new Set(activeData.map(n => n.name));

    graph.clear();

    activeData.forEach(node => {
        graph.addNode(node.name, {
            label: node.name,
            size: Math.sqrt(node.tokens) / 2 + 2,
            color: getCategoryColor(node.category),
            category: node.category,
            centrality: node.centrality,
            tokens: node.tokens,
            file: node.file,
            neighborCount: node.neighborCount,
            x: Math.random(),
            y: Math.random()
        });
    });

    allNodesData.forEach(node => {
        if (activeIds.has(node.name)) {
            node.outbound.forEach(neighbor => {
                if (activeIds.has(neighbor) && !graph.hasEdge(node.name, neighbor)) {
                    graph.addEdge(node.name, neighbor, { size: 1, color: '#333' });
                }
            });
        }
    });

    if (typeof graphologyLibrary !== 'undefined') {
        if (window.FA2Layout) window.FA2Layout.stop();
        window.FA2Layout = new graphologyLibrary.layoutForceAtlas2.FA2Layout(graph, {
            settings: { gravity: 1 }
        });
        window.FA2Layout.start();
        setTimeout(() => window.FA2Layout.stop(), 2000);
    }
}

function getCategoryColor(cat) {
    if (cat === 'priority') return '#ff7b72';
    if (cat === 'vendor') return '#79c0ff';
    if (cat === 'utility') return '#7ee787';
    return '#58a6ff';
}

function showNodeInfo(data) {
    infoElement.innerHTML = `
        <h1>Node Details</h1>
        <div class="node-detail">
            <span class="label">Name</span>
            <span class="value category-${data.category}">${data.label}</span>
        </div>
        <div class="node-detail">
            <span class="label">Category</span>
            <span class="value">${data.category}</span>
        </div>
        <div class="node-detail">
            <span class="label">Centrality (Markov)</span>
            <span class="value">${data.centrality.toFixed(8)}</span>
        </div>
        <div class="node-detail">
            <span class="label">Approx Tokens</span>
            <span class="value">${data.tokens}</span>
        </div>
        <div class="node-detail">
            <span class="label">File</span>
            <span class="value">${data.file}</span>
        </div>
        <div class="node-detail">
            <span class="label">Neighbors</span>
            <span class="value">${data.neighborCount}</span>
        </div>
    `;
}

function resetInfo() {
    infoElement.innerHTML = `
        <h1>Graph Analysis</h1>
        <p>Select a node to see details.</p>
    `;
}

init();
