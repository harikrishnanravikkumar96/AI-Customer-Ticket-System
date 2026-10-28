
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import sqlite3
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from ml_model import predict_solution
from rag_engine import generate_response

DB_PATH = "tickets.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'General',
                priority TEXT NOT NULL DEFAULT 'Medium',
                predicted_solution TEXT NOT NULL DEFAULT '',
                detailed_response TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'Open',
                dynamic_values TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS field_definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                field_type TEXT NOT NULL,
                required INTEGER NOT NULL DEFAULT 0,
                options TEXT NOT NULL DEFAULT '[]'
            )
        """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AI Customer Ticket Response System",
    description="Local ML and RAG customer support assessment",
    lifespan=lifespan,
)


class TicketCreate(BaseModel):
    customer_name: str = Field(min_length=1, max_length=120)
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=5000)
    category: str = "General"
    priority: str = "Medium"
    dynamic_values: dict = {}


class TicketUpdate(BaseModel):
    customer_name: str | None = None
    subject: str | None = None
    description: str | None = None
    category: str | None = None
    priority: str | None = None
    status: str | None = None
    dynamic_values: dict | None = None


class FieldCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=100)
    field_type: str
    required: bool = False
    options: list[str] = []


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AI Customer Ticket System</title>
      <style>
        body{font-family:Arial,sans-serif;max-width:1000px;margin:35px auto;padding:0 18px;color:#222}
        input,textarea,select,button{padding:10px;margin:5px 0;box-sizing:border-box}
        input,textarea,select{width:100%;border:1px solid #bbb;border-radius:5px}
        button{cursor:pointer;background:#185abd;color:white;border:0;border-radius:5px}
        .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
        .ticket{border:1px solid #ddd;border-radius:8px;padding:15px;margin:12px 0}
        .muted{color:#666;font-size:14px}
        @media(max-width:600px){.grid{grid-template-columns:1fr}}
      </style>
    </head>
    <body>
      <h1>AI Customer Ticket Response System</h1>
      <p class="muted">Manage support tickets, predict solutions with local ML, and generate grounded responses using local AI and RAG.</p>
      
      <h2>Manage Custom Fields</h2>
      <form id="fieldForm">
        <div class="grid">
          <div>
            <label>Field label</label>
            <input id="fieldLabel" required maxlength="100"
                   placeholder="e.g. Device Type">
          </div>
          <div>
            <label>Field type</label>
            <select id="fieldType">
              <option value="text">Text</option>
              <option value="dropdown">Dropdown</option>
              <option value="number">Number</option>
              <option value="date">Date</option>
              <option value="checkbox">Checkbox</option>
            </select>
          </div>
        </div>
        <label>Dropdown options (comma-separated)</label>
        <input id="fieldOptions"
               placeholder="Laptop, Desktop, Mobile">
        <label>
          <input id="fieldRequired" type="checkbox" style="width:auto">
          Required field
        </label>
        <button type="submit">Add Custom Field</button>
      </form>
      <p id="fieldMessage" role="status"></p>
      <div id="fieldList"></div>
      <h2>Create ticket</h2>
      <form id="ticketForm">
        <div class="grid">
          <div><label>Customer name</label><input name="customer_name" required></div>
          <div><label>Subject</label><input name="subject" required></div>
          <div><label>Category</label>
            <select name="category"><option>General</option><option>Account</option><option>Payment</option><option>Order</option><option>Technical</option></select>
          </div>
          <div><label>Priority</label>
            <select name="priority"><option>Low</option><option selected>Medium</option><option>High</option><option>Critical</option></select>
          </div>
        </div>
        <label>Description</label><textarea name="description" rows="4" required></textarea>
        <div id="dynamicTicketFields"></div>
        <button type="submit">Create ticket</button>
      </form>
      <p id="message" role="status"></p>
      <h2>Tickets</h2>
      <button onclick="loadTickets()">Refresh tickets</button>
      <div id="tickets">Loading...</div>
      <script>
          async function loadFields() {
          const list = document.getElementById('fieldList');
          const message = document.getElementById('fieldMessage');

          try {
            const res = await fetch('/api/fields');
            if (!res.ok) throw new Error('Could not load custom fields');

            const fields = await res.json();
            
            const container = document.getElementById('dynamicTicketFields');
            container.replaceChildren();

            fields.forEach(field => {
              const wrapper = document.createElement('div');
              wrapper.style.margin = '10px 0';

              const label = document.createElement('label');
              label.textContent = field.label + (field.required ? ' *' : '');

              const input = document.createElement(
                field.field_type === 'dropdown' ? 'select' : 'input'
              );

              input.name = 'dynamic_' + field.name;
              input.dataset.dynamicField = field.name;
              input.dataset.fieldType = field.field_type;
              input.required = Boolean(field.required);

              if (field.field_type === 'dropdown') {
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = 'Select ' + field.label;
                input.appendChild(placeholder);

                (field.options || []).forEach(option => {
                  const opt = document.createElement('option');
                  opt.value = option;
                  opt.textContent = option;
                  input.appendChild(opt);
                });
              } else if (field.field_type === 'checkbox') {
                input.type = 'checkbox';
              } else {
                input.type = field.field_type;
              }

              if (field.field_type === 'checkbox') {
                wrapper.append(input, label);
              } else {
                wrapper.append(label, input);
              }

              container.appendChild(wrapper);
            });

            list.replaceChildren();

            if (fields.length === 0) {
              list.textContent = 'No custom fields created yet.';
              return;
            }

            fields.forEach(field => {
              const item = document.createElement('p');
              const requiredText = field.required ? ' (Required)' : '';
              const optionsText = field.field_type === 'dropdown'
                ? ' — Options: ' + (field.options || []).join(', ')
                : '';

              item.textContent =
                field.label + ' [' + field.field_type + ']' +
                requiredText + optionsText;

              list.appendChild(item);
            });
          } catch (err) {
            message.textContent = 'Error loading fields: ' + err.message;
          }
        }

        document.getElementById('fieldForm').addEventListener(
          'submit', async event => {
            event.preventDefault();

            const label = document.getElementById('fieldLabel').value.trim();
            const fieldType = document.getElementById('fieldType').value;
            const required = document.getElementById('fieldRequired').checked;
            const optionsText = document.getElementById('fieldOptions').value;

            const options = fieldType === 'dropdown'
              ? optionsText.split(',').map(x => x.trim()).filter(Boolean)
              : [];

            const message = document.getElementById('fieldMessage');
            message.textContent = 'Saving custom field...';

            try {
              const res = await fetch('/api/fields', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  name: label,
                  label: label,
                  field_type: fieldType,
                  required: required,
                  options: options
                })
              });

              const result = await res.json();

              if (!res.ok) {
                throw new Error(result.detail || 'Could not create field');
              }

              message.textContent = 'Custom field added successfully.';
              event.target.reset();
              await loadFields();
            } catch (err) {
              message.textContent = 'Error: ' + err.message;
            }
          }
        );

        document.getElementById('fieldType').addEventListener('change', () => {
          const isDropdown =
            document.getElementById('fieldType').value === 'dropdown';
          document.getElementById('fieldOptions').disabled = !isDropdown;
          document.getElementById('fieldOptions').required = isDropdown;
        });

        document.getElementById('fieldOptions').disabled =
          document.getElementById('fieldType').value !== 'dropdown';

        loadFields();
        async function loadTickets(){
          const el=document.getElementById('tickets');
          try{
            const r=await fetch('/api/tickets');
            if(!r.ok) throw new Error(await r.text());
            const tickets=await r.json();
            el.innerHTML='';
            if(!tickets.length){el.textContent='No tickets yet.';return;}
            tickets.forEach(t=>{
              const card=document.createElement('section');
              card.className='ticket';
              const title=document.createElement('h3');
              title.textContent='#'+t.id+' — '+t.subject;
              const details=document.createElement('p');
              details.textContent=t.customer_name+' | '+t.category+' | '+t.priority+' | '+t.status;
              const description=document.createElement('p');
              description.textContent=t.description;
              const status=document.createElement('select');
              ['Open','In Progress','Resolved','Closed'].forEach(s=>{
                const opt=document.createElement('option');
                opt.value=s;opt.textContent=s;opt.selected=s===t.status;
                status.appendChild(opt);
              });
              status.addEventListener('change',async()=>{
                const res=await fetch('/api/tickets/'+t.id,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:status.value})});
                if(!res.ok){alert('Could not update status');loadTickets();}
              });
              const deactivate=document.createElement('button');
              deactivate.textContent='Deactivate ticket';
              deactivate.addEventListener('click',async()=>{
                if(!confirm('Deactivate ticket #'+t.id+'?')) return;
                const res=await fetch('/api/tickets/'+t.id,{method:'DELETE'});
                if(res.ok) loadTickets(); else alert('Could not deactivate ticket');
              });
              card.append(title,details,description,document.createTextNode('Status: '),status,deactivate);
              
              if (t.dynamic_values && Object.keys(t.dynamic_values).length > 0) {
                const customDetails = document.createElement('p');
                customDetails.textContent = 'Custom Fields: ' +
                  Object.entries(t.dynamic_values)
                    .map(([key, value]) => key + ': ' + value)
                    .join(' | ');
                card.appendChild(customDetails);
              }

              if(t.predicted_solution){
                const prediction=document.createElement('p');
                prediction.textContent='ML solution: '+t.predicted_solution;
                card.appendChild(prediction);
              }
              
const responseHeading = document.createElement('h4');
responseHeading.textContent = 'Detailed AI Response';

const responseEditor = document.createElement('textarea');
responseEditor.rows = 5;
responseEditor.style.width = '100%';
responseEditor.maxLength = 10000;
responseEditor.placeholder = 'Generate an AI response or write/edit a response here...';
responseEditor.value = t.detailed_response || '';

const generateButton = document.createElement('button');
generateButton.textContent = 'Generate AI Response';

const saveButton = document.createElement('button');
saveButton.textContent = 'Save Edited Response';
saveButton.style.marginLeft = '8px';

const responseStatus = document.createElement('p');
responseStatus.className = 'muted';
responseStatus.setAttribute('role', 'status');

generateButton.addEventListener('click', async () => {
  generateButton.disabled = true;
  responseStatus.textContent = 'Generating response using local AI...';

  try {
    const res = await fetch(
      '/api/tickets/' + t.id + '/generate-response',
      { method: 'POST' }
    );

    const result = await res.json();

    if (!res.ok) {
      throw new Error(result.detail || 'Generation failed');
    }

    responseEditor.value = result.detailed_response || '';
    responseStatus.textContent =
      'Generated and saved. Knowledge sources: ' +
      (result.sources || []).map(s =>
        typeof s === 'string' ? s : (s.source || s.filename || JSON.stringify(s))
      ).join(', ');
  } catch (err) {
    responseStatus.textContent = 'Error: ' + err.message;
  } finally {
    generateButton.disabled = false;
  }
});

saveButton.addEventListener('click', async () => {
  saveButton.disabled = true;
  responseStatus.textContent = 'Saving response...';

  try {
    const res = await fetch(
      '/api/tickets/' + t.id + '/response',
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detailed_response: responseEditor.value
        })
      }
    );

    const result = await res.json();

    if (!res.ok) {
      throw new Error(result.detail || 'Save failed');
    }

    responseStatus.textContent = 'Edited response saved successfully.';
  } catch (err) {
    responseStatus.textContent = 'Error: ' + err.message;
  } finally {
    saveButton.disabled = false;
  }
});

card.append(
  responseHeading,
  responseEditor,
  generateButton,
  saveButton,
  responseStatus
);

el.appendChild(card);

            });
          }catch(e){el.textContent='Unable to load tickets: '+e.message;}
        }
        document.getElementById('ticketForm').addEventListener('submit',async e=>{
          e.preventDefault();
          const form=e.target;
          
          const formData = new FormData(form);
          const data = Object.fromEntries(formData.entries());
          data.dynamic_values = {};

          document.querySelectorAll('[data-dynamic-field]').forEach(input => {
            const name = input.dataset.dynamicField;

            if (input.dataset.fieldType === 'checkbox') {
              data.dynamic_values[name] = input.checked;
            } else {
              data.dynamic_values[name] = input.value;
            }

            delete data['dynamic_' + name];
          });

          try{
            const r=await fetch('/api/tickets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
            const result=await r.json();
            if(!r.ok) throw new Error(result.detail||'Request failed');
            document.getElementById('message').textContent='Created ticket #'+result.id;
            form.reset();loadTickets();
          }catch(err){document.getElementById('message').textContent='Error: '+err.message;}
        });
        loadTickets();
      </script>
    </body>
    </html>
    """


@app.get("/api/health")
def health():
    return {"status": "ok", "database": DB_PATH}


@app.post("/api/tickets", status_code=201)
def create_ticket(ticket: TicketCreate):
    now = datetime.now(timezone.utc).isoformat()

    # Predict a solution using our locally trained ML model
    predicted_solution = predict_solution(
        ticket.subject,
        ticket.description
    )

    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO tickets
            (customer_name, subject, description, category, priority,
             predicted_solution, dynamic_values, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket.customer_name.strip(),
            ticket.subject.strip(),
            ticket.description.strip(),
            ticket.category,
            ticket.priority,
            predicted_solution,
            json.dumps(ticket.dynamic_values),
            now,
            now,
        ))
        ticket_id = cursor.lastrowid

    return get_ticket(ticket_id)

@app.get("/api/tickets")
def list_tickets():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE is_active=1 ORDER BY id DESC"
        ).fetchall()
    return [dict(row) | {"dynamic_values": json.loads(row["dynamic_values"])}
            for row in rows]


@app.get("/api/tickets/{ticket_id}")
def get_ticket(ticket_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE id=? AND is_active=1",
            (ticket_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    result = dict(row)
    result["dynamic_values"] = json.loads(result["dynamic_values"])
    return result


@app.patch("/api/tickets/{ticket_id}")
def update_ticket(ticket_id: int, update: TicketUpdate):
    data = update.model_dump(exclude_unset=True)
    if not data:
        raise HTTPException(status_code=400, detail="No fields supplied")

    allowed = {
        "customer_name", "subject", "description",
        "category", "priority", "status", "dynamic_values"
    }
    if not set(data).issubset(allowed):
        raise HTTPException(status_code=400, detail="Invalid fields")

    for key in ("customer_name", "subject", "description"):
        if key in data and (data[key] is None or not str(data[key]).strip()):
            raise HTTPException(status_code=400, detail=f"{key} cannot be empty")

    if "status" in data and data["status"] not in {
        "Open", "In Progress", "Resolved", "Closed"
    }:
        raise HTTPException(status_code=400, detail="Invalid status")

    if "dynamic_values" in data:
        data["dynamic_values"] = json.dumps(data["dynamic_values"])

    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    assignments = ", ".join(f"{key}=?" for key in data)
    values = list(data.values()) + [ticket_id]

    with get_db() as conn:
        cursor = conn.execute(
            f"UPDATE tickets SET {assignments} WHERE id=? AND is_active=1",
            values,
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
    return {"message": "Ticket updated successfully"}


@app.delete("/api/tickets/{ticket_id}")
def deactivate_ticket(ticket_id: int):
    with get_db() as conn:
        cursor = conn.execute("""
            UPDATE tickets SET is_active=0, updated_at=?
            WHERE id=? AND is_active=1
        """, (datetime.now(timezone.utc).isoformat(), ticket_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
    return {"message": "Ticket deactivated successfully"}


@app.post("/api/fields", status_code=201)
def create_dynamic_field(field: FieldCreate):
    if field.field_type not in {"text", "dropdown", "number", "date", "checkbox"}:
        raise HTTPException(status_code=400, detail="Unsupported field type")
    if field.field_type == "dropdown" and not field.options:
        raise HTTPException(status_code=400, detail="Dropdown fields need options")

    with get_db() as conn:
        try:
            cursor = conn.execute("""
                INSERT INTO field_definitions (name, label, field_type, required, options)
                VALUES (?, ?, ?, ?, ?)
            """, (
                field.name.strip().lower().replace(" ", "_"),
                field.label.strip(),
                field.field_type,
                int(field.required),
                json.dumps(field.options),
            ))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Field name already exists")
    return {"id": cursor.lastrowid, "message": "Dynamic field created"}


@app.get("/api/fields")
def list_dynamic_fields():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM field_definitions ORDER BY id"
        ).fetchall()
    return [dict(row) | {"options": json.loads(row["options"])} for row in rows]


@app.post("/api/tickets/{ticket_id}/generate-response")
def generate_ticket_response(ticket_id: int):
    ticket = get_ticket(ticket_id)

    try:
        result = generate_response(
            subject=ticket["subject"],
            description=(
                f"Predicted solution: {ticket['predicted_solution']}\n"
                f"Customer issue: {ticket['description']}"
            ),
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local Ollama service unavailable: {exc}",
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate response: {exc}",
        )

    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute(
            """UPDATE tickets
               SET detailed_response=?, updated_at=?
               WHERE id=? AND is_active=1""",
            (result["response"], now, ticket_id),
        )

    return {
        "ticket_id": ticket_id,
        "detailed_response": result["response"],
        "sources": result["sources"],
        "message": "AI response generated and saved successfully",
    }


class ResponseUpdate(BaseModel):
    detailed_response: str = Field(min_length=1, max_length=10000)


@app.patch("/api/tickets/{ticket_id}/response")
def edit_ticket_response(ticket_id: int, update: ResponseUpdate):
    response_text = update.detailed_response.strip()

    if not response_text:
        raise HTTPException(
            status_code=400,
            detail="Response cannot be empty",
        )

    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cursor = conn.execute(
            """UPDATE tickets
               SET detailed_response=?, updated_at=?
               WHERE id=? AND is_active=1""",
            (response_text, now, ticket_id),
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Ticket not found",
            )

    return {
        "ticket_id": ticket_id,
        "detailed_response": response_text,
        "message": "Response saved successfully",
    }
