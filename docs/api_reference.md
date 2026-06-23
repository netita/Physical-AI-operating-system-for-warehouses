# WarehouseGPT API Reference

All API services are built with FastAPI and served behind the `api-gateway` service (default port 8000). The agent service exposes its own endpoints on port 8001 internally.

## Base URLs

| Service | Internal URL | External URL |
|---|---|---|
| API Gateway | `http://api-gateway:8000` | `https://api.warehouse.internal` |
| Agent Service | `http://agent:8001` | `https://agent.warehouse.internal` |
| Digital Twin API | `http://api-gateway:8000/twin` | via gateway |

---

## Authentication

### JWT Authentication

All endpoints (except `/health`, `/metrics`, `/docs`) require a valid JWT Bearer token.

**Obtain a token:**

```http
POST /auth/token
Content-Type: application/json

{
  "username": "operator@warehouse.internal",
  "password": "...",
  "warehouse_id": "wh-001"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "...",
  "warehouse_id": "wh-001",
  "roles": ["operator", "viewer"]
}
```

**Use the token:**

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Token refresh:**

```http
POST /auth/refresh
Authorization: Bearer <refresh_token>
```

### JWT Payload Schema

```json
{
  "sub": "user-uuid-v4",
  "email": "operator@warehouse.internal",
  "warehouse_id": "wh-001",
  "roles": ["operator"],
  "iat": 1720000000,
  "exp": 1720003600,
  "jti": "unique-token-id"
}
```

### Role Permissions

| Role | Permissions |
|---|---|
| `viewer` | GET endpoints only; no mission creation |
| `operator` | All GET + POST mission/agent endpoints |
| `safety_officer` | All operator permissions + override E-stops |
| `admin` | Full access including system configuration |
| `service` | Machine-to-machine; scoped to specific tool calls |

---

## Rate Limiting

Rate limits are enforced per `(user_id, endpoint)` using a Redis sliding-window counter.

| Endpoint Group | Limit | Window |
|---|---|---|
| `/agent/chat` | 60 requests | 1 minute |
| `/agent/chat` (streaming) | 10 concurrent | per user |
| `/twin/*` | 300 requests | 1 minute |
| `/fleet/*` | 120 requests | 1 minute |
| `/safety/*` | 600 requests | 1 minute |
| `/auth/token` | 10 requests | 1 minute |

Rate limit headers are returned on every response:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1720003600
```

When the limit is exceeded:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 23

{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Retry after 23 seconds.",
  "retry_after_seconds": 23
}
```

---

## Common Response Schema

All responses follow this envelope:

```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "req-uuid-v4",
    "timestamp": "2026-06-19T10:30:00Z",
    "latency_ms": 42
  }
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Forklift fk-007 not found in warehouse wh-001",
    "details": { "forklift_id": "fk-007", "warehouse_id": "wh-001" }
  },
  "meta": {
    "request_id": "req-uuid-v4",
    "timestamp": "2026-06-19T10:30:00Z"
  }
}
```

---

## Digital Twin API

### GET /twin/state

Returns the current full state of the digital twin.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warehouse_id` | string | from JWT | Target warehouse |
| `include_predictions` | bool | false | Include world model predictions |
| `horizon_seconds` | int | 5 | Prediction horizon if include_predictions=true |

**Response:**

```json
{
  "success": true,
  "data": {
    "warehouse_id": "wh-001",
    "snapshot_ts": "2026-06-19T10:30:00.123Z",
    "world_model_confidence": 0.94,
    "forklifts": [
      {
        "id": "fk-001",
        "pose": {
          "x": 12.4, "y": 8.7, "z": 0.0,
          "yaw_deg": 45.0
        },
        "velocity": { "linear_mps": 1.2, "angular_rps": 0.0 },
        "load": {
          "carrying": true,
          "pallet_id": "plt-8823",
          "weight_kg": 450.0
        },
        "battery_pct": 78,
        "status": "executing_mission",
        "mission_id": "msn-4421",
        "safety_stop_active": false
      }
    ],
    "workers": [
      {
        "id": "wkr-003",
        "pose": { "x": 15.1, "y": 9.2, "z": 0.0, "yaw_deg": 180.0 },
        "zone": "A3",
        "ppe_detected": ["hard_hat", "vest", "gloves"],
        "ppe_compliant": true,
        "nearest_vehicle_distance_m": 4.8
      }
    ],
    "zones": [
      {
        "id": "zone-A3",
        "name": "A3",
        "type": "storage",
        "occupancy_pct": 72.0,
        "active_vehicles": 1,
        "active_workers": 1,
        "safety_alert": null
      }
    ],
    "inventory_summary": {
      "total_pallets": 1240,
      "pallets_in_transit": 8,
      "dock_doors_open": ["dock-5", "dock-7"]
    },
    "predictions": {
      "horizon_seconds": 5,
      "occupancy_grid": "base64-encoded-3D-array",
      "forklift_trajectories": [
        {
          "forklift_id": "fk-001",
          "waypoints": [
            { "t_offset_s": 1.0, "x": 13.0, "y": 8.9 },
            { "t_offset_s": 5.0, "x": 20.0, "y": 10.2 }
          ]
        }
      ]
    }
  }
}
```

---

### GET /twin/state/forklift/{forklift_id}

Returns state and telemetry for a single forklift.

**Path Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `forklift_id` | string | Forklift identifier (e.g., `fk-001`) |

**Response:** Single forklift object (see forklift schema above).

---

### GET /twin/history

Returns historical state snapshots for replay and analysis.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_ts` | ISO 8601 | required | Start of history window |
| `end_ts` | ISO 8601 | required | End of history window |
| `entity_type` | string | all | Filter: `forklift`, `worker`, `zone` |
| `entity_id` | string | all | Filter by specific entity |
| `interval_seconds` | int | 10 | Snapshot interval in output |

**Response:**

```json
{
  "success": true,
  "data": {
    "snapshots": [
      {
        "ts": "2026-06-19T08:00:00Z",
        "entities": [ ... ]
      }
    ],
    "total_snapshots": 720,
    "interval_seconds": 10
  }
}
```

---

### POST /twin/calibrate

Triggers a calibration run to align digital twin with real-world sensor readings.

**Request Body:**

```json
{
  "calibration_type": "pose_correction",
  "reference_points": [
    { "marker_id": "qr-A3-NW", "real_x": 10.0, "real_y": 5.0 },
    { "marker_id": "qr-dock7-entrance", "real_x": 80.0, "real_y": 2.0 }
  ],
  "auto_apply": true
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "calibration_id": "cal-2026-0619-001",
    "status": "running",
    "estimated_completion_seconds": 30,
    "previous_drift_m": 0.08,
    "applied": false
  }
}
```

---

### WebSocket /twin/stream

Real-time digital twin state stream over WebSocket.

**Connection:**

```javascript
const ws = new WebSocket('wss://api.warehouse.internal/twin/stream?token=<jwt>');
```

**Server push messages:**

```json
{
  "type": "state_delta",
  "ts": "2026-06-19T10:30:00.500Z",
  "changes": [
    {
      "entity_type": "forklift",
      "entity_id": "fk-001",
      "field": "pose",
      "value": { "x": 12.6, "y": 8.8, "z": 0.0, "yaw_deg": 46.2 }
    }
  ]
}
```

```json
{
  "type": "safety_alert",
  "ts": "2026-06-19T10:30:05.123Z",
  "alert": {
    "alert_id": "alt-9981",
    "severity": "high",
    "category": "near_miss",
    "description": "Forklift fk-003 within 1.2m of worker wkr-007 in zone B2",
    "entities": ["fk-003", "wkr-007"],
    "zone": "B2",
    "recommended_action": "emergency_stop_fk003"
  }
}
```

---

## WarehouseGPT Agent API

### POST /agent/chat

Send a natural language message to the WarehouseGPT agent.

**Request Body:**

```json
{
  "message": "Move all pallets from zone A3 to dock 7 before shift end",
  "conversation_id": "conv-uuid-v4",
  "warehouse_id": "wh-001",
  "context": {
    "current_shift": "day",
    "shift_end_ts": "2026-06-19T16:00:00Z",
    "operator_id": "op-042"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | Natural language instruction or question |
| `conversation_id` | string | no | Omit to start a new conversation |
| `warehouse_id` | string | no | Defaults to JWT warehouse |
| `context` | object | no | Optional operator context metadata |

**Response:**

```json
{
  "success": true,
  "data": {
    "conversation_id": "conv-uuid-v4",
    "response": "Understood. I've identified 42 pallets in zone A3. I'm assigning forklifts fk-001, fk-003, and fk-007 to transport them to dock 7. Estimated completion: 14:23 (1h37m before shift end). Mission ID: msn-4421. I'll notify you if any delays arise.",
    "tool_calls_made": [
      { "tool": "query_scene_graph", "args": { "zone": "A3" }, "result_summary": "42 pallets found" },
      { "tool": "get_twin_state", "args": {}, "result_summary": "3 forklifts available" },
      { "tool": "create_mission", "args": { "pallets": 42, "target": "dock7" }, "result_summary": "msn-4421 created" }
    ],
    "missions_created": ["msn-4421"],
    "thinking_tokens": 1420,
    "output_tokens": 98,
    "latency_ms": 2340
  }
}
```

---

### POST /agent/chat/stream

Streaming version of the chat endpoint using Server-Sent Events (SSE).

**Request Body:** Same as `/agent/chat`.

**Response:** `Content-Type: text/event-stream`

```
data: {"type": "thinking", "content": "Let me check the current state of zone A3..."}

data: {"type": "tool_call", "tool": "query_scene_graph", "args": {"zone": "A3"}}

data: {"type": "tool_result", "tool": "query_scene_graph", "summary": "42 pallets found"}

data: {"type": "content", "delta": "Understood. I've identified 42 pallets"}

data: {"type": "content", "delta": " in zone A3."}

data: {"type": "done", "conversation_id": "conv-uuid-v4", "missions_created": ["msn-4421"]}
```

---

### GET /agent/conversation/{conversation_id}

Retrieve the full conversation history.

**Response:**

```json
{
  "success": true,
  "data": {
    "conversation_id": "conv-uuid-v4",
    "warehouse_id": "wh-001",
    "operator_id": "op-042",
    "started_at": "2026-06-19T10:00:00Z",
    "last_active_at": "2026-06-19T10:30:00Z",
    "messages": [
      {
        "role": "user",
        "content": "Move all pallets from zone A3 to dock 7 before shift end",
        "ts": "2026-06-19T10:30:00Z"
      },
      {
        "role": "assistant",
        "content": "Understood. I've identified 42 pallets...",
        "ts": "2026-06-19T10:30:02Z",
        "tool_calls": [ ... ]
      }
    ]
  }
}
```

---

### POST /agent/tool/run

Directly invoke an agent tool without NL parsing (for programmatic integrations).

**Request Body:**

```json
{
  "tool": "dispatch_forklift",
  "args": {
    "forklift_id": "fk-001",
    "task": "pick",
    "pallet_id": "plt-8823",
    "source_location": { "zone": "A3", "shelf": "A3-07", "slot": 2 },
    "destination": { "zone": "dock", "dock_door": 7 }
  }
}
```

**Available Tools:**

| Tool | Description |
|---|---|
| `query_scene_graph` | Query Neo4j spatial graph (zones, shelves, pallets, vehicles) |
| `dispatch_forklift` | Assign a specific task to a forklift |
| `run_safety_check` | Run a safety assessment for a zone or entity |
| `get_inventory` | Query WMS inventory state |
| `create_mission` | Create a multi-step mission |
| `query_rag` | Semantic search over SOPs and knowledge base |
| `get_twin_state` | Retrieve current digital twin state |
| `get_mission_status` | Query status of an existing mission |
| `cancel_mission` | Cancel a running mission |
| `trigger_e_stop` | Trigger emergency stop on vehicle(s) |

---

### GET /agent/missions

List missions with optional filters.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `status` | string | all | Filter: `pending`, `running`, `completed`, `failed`, `cancelled` |
| `warehouse_id` | string | from JWT | Target warehouse |
| `created_after` | ISO 8601 | - | Filter by creation time |
| `limit` | int | 50 | Max results |
| `offset` | int | 0 | Pagination offset |

**Response:**

```json
{
  "success": true,
  "data": {
    "missions": [
      {
        "mission_id": "msn-4421",
        "status": "running",
        "progress_pct": 42,
        "created_at": "2026-06-19T10:30:02Z",
        "estimated_completion": "2026-06-19T14:23:00Z",
        "sub_tasks_total": 42,
        "sub_tasks_completed": 18,
        "sub_tasks_failed": 0,
        "assigned_forklifts": ["fk-001", "fk-003", "fk-007"],
        "created_by": {
          "type": "agent",
          "conversation_id": "conv-uuid-v4",
          "operator_id": "op-042"
        }
      }
    ],
    "total": 1,
    "limit": 50,
    "offset": 0
  }
}
```

---

### GET /agent/missions/{mission_id}

Get detailed status of a specific mission.

---

### DELETE /agent/missions/{mission_id}

Cancel a running or pending mission.

**Request Body:**

```json
{
  "reason": "shift_change",
  "reassign_tasks": false
}
```

---

## Fleet Manager API

### GET /fleet/forklifts

List all forklifts and their current status.

**Response:**

```json
{
  "success": true,
  "data": {
    "forklifts": [
      {
        "id": "fk-001",
        "model": "Crown FC5200",
        "status": "executing_mission",
        "mission_id": "msn-4421",
        "battery_pct": 78,
        "load_kg": 450,
        "max_load_kg": 1800,
        "pose": { "x": 12.4, "y": 8.7, "yaw_deg": 45.0 },
        "zone": "A3",
        "last_seen_ts": "2026-06-19T10:30:00.100Z",
        "health": {
          "motor_temp_c": 42,
          "hydraulic_pressure_bar": 180,
          "faults": []
        }
      }
    ],
    "summary": {
      "total": 12,
      "available": 5,
      "executing_mission": 4,
      "charging": 2,
      "fault": 1
    }
  }
}
```

---

### POST /fleet/forklifts/{forklift_id}/command

Send a direct command to a forklift (requires `operator` role).

**Request Body:**

```json
{
  "command": "go_to_charging_station",
  "args": {
    "station_id": "cs-02",
    "priority": "normal"
  }
}
```

**Available Commands:** `go_to_charging_station`, `go_to_home_position`, `emergency_stop`, `resume`, `cancel_current_task`

---

### POST /fleet/emergency_stop

Trigger emergency stop on all or specific vehicles (requires `safety_officer` role).

**Request Body:**

```json
{
  "scope": "zone",
  "zone_id": "B2",
  "reason": "worker_in_danger",
  "alert_id": "alt-9981"
}
```

**Scope options:** `all`, `zone`, `vehicle_list`

---

## Safety API

### GET /safety/alerts

List active and recent safety alerts.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `status` | string | `active` | `active`, `acknowledged`, `resolved`, `all` |
| `severity` | string | all | `critical`, `high`, `medium`, `low` |
| `zone_id` | string | all | Filter by zone |
| `limit` | int | 100 | Max results |

**Response:**

```json
{
  "success": true,
  "data": {
    "alerts": [
      {
        "alert_id": "alt-9981",
        "severity": "high",
        "category": "near_miss",
        "status": "active",
        "detected_at": "2026-06-19T10:30:05Z",
        "description": "Forklift fk-003 within 1.2m of worker wkr-007 in zone B2",
        "entities": [
          { "type": "forklift", "id": "fk-003" },
          { "type": "worker", "id": "wkr-007" }
        ],
        "zone_id": "B2",
        "detector": "near_miss",
        "confidence": 0.97,
        "frame_uri": "s3://warehousegpt/alerts/alt-9981/frame.jpg",
        "actions_taken": ["auto_speed_reduction_fk003"],
        "recommended_actions": ["emergency_stop_fk003", "alert_supervisor"]
      }
    ],
    "active_count": 1,
    "acknowledged_count": 3
  }
}
```

---

### POST /safety/alerts/{alert_id}/acknowledge

Acknowledge a safety alert.

**Request Body:**

```json
{
  "acknowledged_by": "op-042",
  "action_taken": "emergency_stop_fk003",
  "notes": "Stopped forklift, escorted worker to safe zone"
}
```

---

### GET /safety/metrics

Safety KPIs and statistics.

**Query Parameters:** `period` (day/week/month), `warehouse_id`

**Response:**

```json
{
  "success": true,
  "data": {
    "period": "week",
    "near_misses": 3,
    "fire_detections": 0,
    "zone_violations": 12,
    "ppe_violations": 7,
    "e_stops_triggered": 1,
    "days_without_incident": 14,
    "mttr_minutes": 4.2,
    "osha_compliance_score": 98.2,
    "trend": "improving"
  }
}
```

---

## Knowledge & Inventory API

### GET /inventory/pallets

Query warehouse inventory.

**Query Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `zone` | string | Filter by zone |
| `sku` | string | Filter by SKU |
| `status` | string | `stored`, `in_transit`, `at_dock`, `all` |
| `limit` | int | Max results (default 100) |

**Response:**

```json
{
  "success": true,
  "data": {
    "pallets": [
      {
        "pallet_id": "plt-8823",
        "sku": "ITEM-4421-XL",
        "description": "Widget XL 48-pack",
        "weight_kg": 450,
        "dimensions_cm": { "l": 120, "w": 100, "h": 150 },
        "location": {
          "zone": "A3",
          "shelf": "A3-07",
          "slot": 2,
          "x": 14.2, "y": 9.1
        },
        "status": "in_transit",
        "forklift_carrying": "fk-001",
        "arrival_ts": "2026-06-18T14:00:00Z",
        "expiry_ts": null
      }
    ],
    "total": 1240,
    "returned": 1
  }
}
```

---

### POST /knowledge/search

Semantic search over the RAG knowledge base (SOPs, safety manuals, equipment docs).

**Request Body:**

```json
{
  "query": "maximum pallet weight for zone A shelving",
  "top_k": 5,
  "collections": ["safety_sops", "equipment_manuals", "warehouse_rules"]
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "document_id": "sop-safety-001",
        "title": "Zone A Storage Weight Limits",
        "excerpt": "Zone A shelving is rated for a maximum of 1,200 kg per shelf level. Individual pallets must not exceed 800 kg. Verify pallet weight on the WMS before placement.",
        "score": 0.94,
        "source": "safety_sops",
        "url": "confluence://spaces/WHS/pages/4421"
      }
    ]
  }
}
```

---

## System API

### GET /health

Health check endpoint (no authentication required).

**Response:**

```json
{
  "status": "healthy",
  "services": {
    "postgres": "healthy",
    "redis": "healthy",
    "neo4j": "healthy",
    "chromadb": "healthy",
    "triton": "healthy"
  },
  "version": "0.1.0",
  "uptime_seconds": 86400
}
```

### GET /metrics

Prometheus metrics endpoint (no authentication, restricted to internal network).

### GET /docs

OpenAPI interactive documentation (Swagger UI).

### GET /redoc

OpenAPI reference documentation (ReDoc).

---

## Error Codes

| Code | HTTP Status | Description |
|---|---|---|
| `AUTH_TOKEN_MISSING` | 401 | No Authorization header |
| `AUTH_TOKEN_INVALID` | 401 | JWT signature invalid or malformed |
| `AUTH_TOKEN_EXPIRED` | 401 | JWT past expiry |
| `AUTH_INSUFFICIENT_ROLE` | 403 | User lacks required role |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `RESOURCE_NOT_FOUND` | 404 | Entity does not exist |
| `VALIDATION_ERROR` | 422 | Request body schema violation |
| `FORKLIFT_UNAVAILABLE` | 409 | Forklift is not in an assignable state |
| `MISSION_CONFLICT` | 409 | Mission conflicts with existing assignment |
| `TWIN_DESYNC` | 503 | Digital twin out of sync with real world |
| `AGENT_OVERLOADED` | 503 | Agent service at capacity |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
