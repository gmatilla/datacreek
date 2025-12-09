import { useEffect, useState, useRef, useMemo } from "react";
import { useParams } from "react-router-dom";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@radix-ui/react-tabs";
import * as Checkbox from "@radix-ui/react-checkbox";
import { CheckIcon } from "@radix-ui/react-icons";
import ForceGraph3D from "react-force-graph-3d";
import { Card, CardHeader, CardContent, Progress, Button } from "./ui";

export default function DatasetDetail() {
  const { name } = useParams();
  const [info, setInfo] = useState(null);
  const [graph, setGraph] = useState(null);
  const [content, setContent] = useState([]);
  const [filters, setFilters] = useState({ document: true, chunk: true });
  const [sourceFilter, setSourceFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [curQuery, setCurQuery] = useState("");
  const [conflicts, setConflicts] = useState([]);
  const [pruneSources, setPruneSources] = useState("");
  const [entityId, setEntityId] = useState("");
  const graphRef = useRef();

  useEffect(() => {
    fetch(`/api/datasets/${name}`)
      .then((r) => r.json())
      .then(setInfo);
    fetch(`/datasets/${name}/graph`)
      .then((r) => r.json())
      .then(setGraph);
    fetch(`/api/datasets/${name}/content`)
      .then((r) => r.json())
      .then(setContent);
  }, [name]);

  const filteredGraph = useMemo(() => {
    if (!graph) return null;
    const sf = sourceFilter.toLowerCase();
    const nodeSet = new Set(
      graph.nodes
        .filter(
          (n) =>
            filters[n.type] &&
            (!sf || String(n.source || "").toLowerCase().includes(sf)),
        )
        .map((n) => n.id),
    );
    return {
      nodes: graph.nodes.filter((n) => nodeSet.has(n.id)),
      edges: graph.edges.filter(
        (e) => nodeSet.has(e.source) && nodeSet.has(e.target),
      ),
    };
  }, [graph, filters, sourceFilter]);

  const curChunks = useMemo(() => {
    const all = content.flatMap((d) => d.chunks);
    if (!curQuery) return all;
    const q = curQuery.toLowerCase();
    return all.filter((c) => c.text.toLowerCase().includes(q));
  }, [content, curQuery]);

  async function deleteChunk(id) {
    await fetch(`/api/datasets/${name}/chunks/${id}`, { method: "DELETE" });
    setContent((docs) =>
      docs.map((doc) => ({
        ...doc,
        chunks: doc.chunks.filter((c) => c.id !== id),
      })),
    );
    fetch(`/api/datasets/${name}`)
      .then((r) => r.json())
      .then(setInfo);
  }

  async function deduplicate() {
    const res = await fetch(`/api/datasets/${name}/deduplicate`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
    fetch(`/api/datasets/${name}/content`).then(r => r.json()).then(setContent)
  }

  async function prune() {
    if (!pruneSources.trim()) return
    const body = { sources: pruneSources.split(',').map(s => s.trim()).filter(Boolean) }
    const res = await fetch(`/api/datasets/${name}/prune`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) return
    setPruneSources('')
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
    fetch(`/api/datasets/${name}/content`).then(r => r.json()).then(setContent)
    fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
  }

  async function cleanChunks() {
    const res = await fetch(`/api/datasets/${name}/clean_chunks`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
    fetch(`/api/datasets/${name}/content`).then(r => r.json()).then(setContent)
  }

  async function normalizeDates() {
    const res = await fetch(`/api/datasets/${name}/normalize_dates`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
    fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
  }

  async function enrichWikidata() {
    if (!entityId.trim()) return
    const res = await fetch(`/api/datasets/${name}/enrich_entity/${entityId}`, { method: 'POST' })
    if (res.ok) {
      setEntityId('')
      fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
    }
  }

  async function enrichDbpedia() {
    if (!entityId.trim()) return
    const res = await fetch(`/api/datasets/${name}/enrich_entity_dbpedia/${entityId}`, { method: 'POST' })
    if (res.ok) {
      setEntityId('')
      fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
    }
  }

  async function runOp(op) {
    const res = await fetch(`/api/datasets/${name}/${op}`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
  }

  async function extractFacts() {
    const res = await fetch(`/api/datasets/${name}/extract_facts`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
  }

  async function checkConflicts() {
    const res = await fetch(`/api/datasets/${name}/conflicts`)
    if (!res.ok) return
    setConflicts(await res.json())
  }

  async function markConflicts() {
    const res = await fetch(`/api/datasets/${name}/mark_conflicts`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
  }

  async function validateCoherence() {
    const res = await fetch(`/api/datasets/${name}/validate`, { method: 'POST' })
    if (!res.ok) return
    fetch(`/datasets/${name}/graph`).then(r => r.json()).then(setGraph)
    fetch(`/api/datasets/${name}`).then(r => r.json()).then(setInfo)
  }

  async function graphSearch() {
    const params = new URLSearchParams({ q: searchQuery, hops: 1 })
    const res = await fetch(`/api/datasets/${name}/search_links?${params}`)
    if (!res.ok) return
    setSearchResults(await res.json())
  }

  async function exportDataset() {
    const res = await fetch(`/api/datasets/${name}/export`);
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.json`;
    a.click();
    URL.revokeObjectURL(url);
    fetch(`/api/datasets/${name}`)
      .then((r) => r.json())
      .then(setInfo);
  }

  if (!info) return <p>Loading...</p>;

  return (
    <div>
      <h2 className="text-xl font-bold mb-4">{name}</h2>
      <Tabs defaultValue="content">
        <TabsList className="flex gap-2 mb-4">
          <TabsTrigger value="content">Content</TabsTrigger>
          {info.stage >= 1 && (
            <TabsTrigger value="graph">Knowledge Graph</TabsTrigger>
          )}
          {info.stage >= 2 && (
            <TabsTrigger value="curation">Curation</TabsTrigger>
          )}
        </TabsList>
        <TabsContent value="content">
          <Card className="mb-4">
            <CardHeader>Info</CardHeader>
            <CardContent>
              <dl className="grid grid-cols-2 gap-y-2 text-sm">
                <dt className="font-medium">ID</dt>
                <dd>{info.id}</dd>
                <dt className="font-medium">Type</dt>
                <dd>{info.type}</dd>
                <dt className="font-medium">Created</dt>
                <dd>{new Date(info.created_at).toLocaleString()}</dd>
                <dt className="font-medium">Documents</dt>
                <dd>{info.num_documents}</dd>
                <dt className="font-medium">Chunks</dt>
                <dd>{info.num_chunks}</dd>
                {info.num_facts !== undefined && (
                  <>
                    <dt className="font-medium">Facts</dt>
                    <dd>{info.num_facts}</dd>
                  </>
                )}
                <dt className="font-medium">Size</dt>
                <dd>{info.size}</dd>
              </dl>
            </CardContent>
          </Card>
          <Card className="mb-4">
            <CardHeader>Content</CardHeader>
            <CardContent>
              {content.length ? (
                <ul className="space-y-2 text-sm">
                  {content.map((doc) => (
                    <li key={doc.id} className="border rounded p-2">
                      <div className="font-medium mb-1">{doc.id}</div>
                      <ul className="list-disc list-inside space-y-1">
                        {doc.chunks.map((c) => (
                          <li key={c.id}>{c.text.slice(0, 80)}</li>
                        ))}
                      </ul>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted">No content</p>
              )}
            </CardContent>
          </Card>
          <Card className="mb-4">
            <CardHeader>Quality</CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 mb-2">
                <Progress value={info.quality} className="flex-1" />
                <span className="text-sm">{info.quality}</span>
              </div>
              {info.tips.length > 0 && (
                <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                  {info.tips.map((tip, i) => (
                    <li key={i}>{tip}</li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
          {info.versions.length > 0 && (
            <Card className="mb-4">
              <CardHeader>Generations</CardHeader>
              <CardContent>
                <ul className="text-sm space-y-1">
                  {info.versions.map((v, i) => (
                    <li key={i} className="border-b last:border-none pb-1">
                      <div className="font-medium">v{i + 1}</div>
                      <div className="text-xs text-gray-500">
                        {new Date(v.time).toLocaleString()}
                      </div>
                      {Object.keys(v.params || {}).length > 0 && (
                        <pre className="text-xs bg-gray-50 p-1 rounded mt-1 whitespace-pre-wrap">
                          {JSON.stringify(v.params, null, 2)}
                        </pre>
                      )}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader>History</CardHeader>
            <CardContent>
              {info.history.length ? (
                <ul className="text-sm space-y-1">
                  {info.history.map((h, i) => (
                    <li key={i}>{h}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted">No history</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="graph">
          {info.stage < 1 ? (
            <p className="text-sm text-muted">
              Ingest documents to view the graph.
            </p>
          ) : (
            filteredGraph && (
              <div className="space-y-2">
                <div className="flex gap-4 items-center text-sm">
                  {["document", "chunk"].map((t) => (
                    <label key={t} className="flex items-center gap-1">
                      <Checkbox.Root
                        className="h-4 w-4 border rounded"
                        checked={filters[t]}
                        onCheckedChange={(v) =>
                          setFilters((f) => ({ ...f, [t]: v }))
                        }
                      >
                        <Checkbox.Indicator className="flex items-center justify-center text-white bg-indigo-600">
                          <CheckIcon className="w-3 h-3" />
                        </Checkbox.Indicator>
                      </Checkbox.Root>
                      {t}
                    </label>
                  ))}
                </div>
                <input
                  className="border rounded p-2 text-sm"
                  placeholder="Filter by source"
                  value={sourceFilter}
                  onChange={(e) => setSourceFilter(e.target.value)}
                />
                <div className="flex gap-2 mt-2">
                  <input
                    className="border rounded p-2 text-sm flex-1"
                    placeholder="Search graph"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <Button type="button" onClick={graphSearch}>
                    Search
                  </Button>
                </div>
                {searchResults.length > 0 && (
                  <ul className="text-xs max-h-32 overflow-auto list-disc list-inside mt-2">
                    {searchResults.map((r) => (
                      <li key={r.id}>{r.text.slice(0, 80)}</li>
                    ))}
                  </ul>
                )}
                {conflicts.length > 0 && (
                  <ul className="text-xs max-h-32 overflow-auto list-disc list-inside mt-2">
                    {conflicts.map((c, i) => (
                      <li key={i}>{c[0]} {c[1]} -> {Object.keys(c[2]).join(', ')}</li>
                    ))}
                  </ul>
                )}
                <div className="h-96">
                  <ForceGraph3D
                    ref={graphRef}
                    graphData={filteredGraph}
                    nodeLabel={(n) => `${n.id} (${n.type})`}
                    nodeAutoColorBy="type"
                  />
                </div>
                <div className="flex gap-2 text-sm">
                  <Button type="button" onClick={() => runOp('consolidate')}>Consolidate Schema</Button>
                  <Button type="button" onClick={() => runOp('communities')}>Detect Communities</Button>
                  <Button type="button" onClick={() => runOp('summaries')}>Summarize</Button>
                  <Button type="button" onClick={() => runOp('entity_groups')}>Cluster Entities</Button>
                  <Button type="button" onClick={() => runOp('entity_group_summaries')}>Summarize Entities</Button>
                  <Button type="button" onClick={() => runOp('trust')}>Score Trust</Button>
                  <Button type="button" onClick={() => runOp('similarity')}>Link Similar</Button>
                  <Button type="button" onClick={() => runOp('co_mentions')}>Link Co-Mentions</Button>
                  <Button type="button" onClick={() => runOp('doc_co_mentions')}>Link Docs</Button>
                  <Button type="button" onClick={() => runOp('section_co_mentions')}>Link Sections</Button>
                  <Button type="button" onClick={() => runOp('author_org_links')}>Author Orgs</Button>
                  <Button type="button" onClick={() => runOp('resolve_entities')}>Resolve Entities</Button>
                  <Button type="button" onClick={() => runOp('predict_links')}>Predict Links</Button>
                  <Button type="button" onClick={() => runOp('predict_links?graph=true')}>Graph Link Prediction</Button>
                  <Button type="button" onClick={() => runOp('centrality')}>Centrality</Button>
                  <Button type="button" onClick={() => runOp('graph_embeddings')}>Graph Embeddings</Button>
                  <Button type="button" onClick={extractFacts}>Extract Facts</Button>
                  <Button type="button" onClick={checkConflicts}>View Conflicts</Button>
                  <Button type="button" onClick={markConflicts}>Mark Conflicts</Button>
                  <Button type="button" onClick={validateCoherence}>Validate</Button>
                </div>
                <div className="flex gap-2 items-center mt-2 text-sm">
                  <input
                    className="border rounded p-1 text-sm"
                    placeholder="entity id"
                    value={entityId}
                    onChange={(e) => setEntityId(e.target.value)}
                  />
                  <Button type="button" onClick={enrichWikidata}>Enrich Wikidata</Button>
                  <Button type="button" onClick={enrichDbpedia}>Enrich DBpedia</Button>
                </div>
              </div>
            )
          )}
        </TabsContent>
        <TabsContent value="curation">
          {info.stage < 2 ? (
            <p className="text-sm text-muted">
              Generate dataset before curating.
            </p>
          ) : (
            <Card>
              <CardHeader>Curation</CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-2 justify-end">
                  <input
                    className="border rounded p-1 text-sm"
                    placeholder="sources comma-separated"
                    value={pruneSources}
                    onChange={(e) => setPruneSources(e.target.value)}
                  />
                  <Button type="button" onClick={prune}>Prune</Button>
                  <Button type="button" onClick={deduplicate}>Deduplicate</Button>
                  <Button type="button" onClick={cleanChunks}>Clean Text</Button>
                  <Button type="button" onClick={normalizeDates}>Normalize Dates</Button>
                </div>
                <input
                  className="border rounded p-2 w-full text-sm"
                  placeholder="Filter chunks"
                  value={curQuery}
                  onChange={(e) => setCurQuery(e.target.value)}
                />
                <ul className="space-y-2 max-h-64 overflow-auto text-sm">
                  {curChunks.map((c) => (
                    <li
                      key={c.id}
                      className="border rounded p-2 flex justify-between gap-2"
                    >
                      <span className="flex-1">{c.text.slice(0, 80)}</span>
                      <button
                        className="text-red-600 text-xs"
                        onClick={() => deleteChunk(c.id)}
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                  {curChunks.length === 0 && (
                    <li className="text-gray-500">No chunks</li>
                  )}
                </ul>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
      {info.stage >= 3 && (
        <div className="mt-4">
          <Button onClick={exportDataset}>Export Dataset</Button>
        </div>
      )}
    </div>
  );
}
