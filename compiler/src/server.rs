// ═══════════════════════════════════════════════════════════════════════
// QOMN v1.1 — HTTP Server Mode
// Expone un programa .crys como REST API.
// Endpoints:
//   GET  /health         → status
//   GET  /routes         → tabla de rutas activa
//   POST /query          → enrutar consulta {"q": "Manning n=0.013..."}
//   POST /eval           → evaluar expresión {"expr": "igv(50000.0)"}
//   GET  /crystals       → crystals registrados
// ═══════════════════════════════════════════════════════════════════════

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use crate::vm::Vm;
use crate::ast::Program;
use crate::lexer::Lexer;
use crate::parser::Parser;

pub struct CrysServer {
    vm:      Arc<Mutex<Vm>>,
    prog:    Program,
    port:    u16,
}

impl CrysServer {
    pub fn new(vm: Vm, prog: Program, port: u16) -> Self {
        Self { vm: Arc::new(Mutex::new(vm)), prog, port }
    }

    pub fn run(&self) {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr)
            .unwrap_or_else(|e| { eprintln!("Bind error: {}", e); std::process::exit(1) });

        println!("  QOMN server listening on {}", addr);
        println!("  Endpoints: GET /health  POST /query  POST /eval  GET /routes");

        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let vm   = Arc::clone(&self.vm);
                    let prog = self.prog.clone();
                    std::thread::spawn(move || handle_conn(s, vm, prog));
                }
                Err(e) => eprintln!("Connection error: {}", e),
            }
        }
    }
}

fn handle_conn(mut stream: TcpStream, vm: Arc<Mutex<Vm>>, _prog: Program) {
    use std::io::Read;
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));

    // Read entire request at once
    let mut raw = vec![0u8; 8192];
    let n = stream.read(&mut raw).unwrap_or(0);
    let request = String::from_utf8_lossy(&raw[..n]).to_string();

    // Split headers and body at \r\n\r\n
    let (headers_part, body) = if let Some(pos) = request.find("\r\n\r\n") {
        (&request[..pos], request[pos+4..].to_string())
    } else {
        (request.as_str(), String::new())
    };

    let mut hdr_lines = headers_part.lines();

    // Parse request line
    let request_line = hdr_lines.next().unwrap_or("");
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 { return; }
    let method = parts[0];
    let path   = parts[1];

    let body = body.trim_matches(char::from(0)).to_string();
    let (status, json) = route_request(method, path, &body, &vm);

    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
        status, json.len(), json
    );
    let _ = stream.write_all(response.as_bytes());
}

fn route_request(method: &str, path: &str, body: &str, vm: &Arc<Mutex<Vm>>) -> (&'static str, String) {
    match (method, path) {
        ("GET", "/health") => {
            ("200 OK", r#"{"status":"ok","lang":"QOMN","version":"1.0"}"#.into())
        }

        ("GET", "/routes") => {
            ("200 OK", r#"{"routes":"see loaded .crys program"}"#.into())
        }

        ("GET", "/crystals") => {
            ("200 OK", r#"{"crystals":"see loaded .crys program"}"#.into())
        }

        ("POST", "/query") => {
            // {"q": "Manning n=0.013 R=0.5 S=0.001"}
            let q = extract_json_str(body, "q").unwrap_or_default();
            let mut vm = vm.lock().unwrap();
            match vm.query(&q) {
                Ok(result) => {
                    let escaped = result.replace('"', "\\\"");
                    ("200 OK", format!(r#"{{"ok":true,"query":"{}","result":"{}"}}"#, q, escaped))
                }
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
            }
        }

        ("POST", "/eval") => {
            // {"expr": "igv(50000.0)"}
            let expr_src = extract_json_str(body, "expr").unwrap_or_default();
            let src = format!("let __result = {}\n", expr_src);
            let mut lexer  = Lexer::new(&src);
            let tokens     = lexer.tokenize();
            let mut parser = Parser::new(tokens);
            match parser.parse() {
                Ok(prog) => {
                    let mut vm = vm.lock().unwrap();
                    match vm.run(&prog) {
                        Ok(out) => {
                            let val = out.join(", ");
                            let escaped = val.replace('"', "\\\"");
                            ("200 OK", format!(r#"{{"ok":true,"expr":"{}","result":"{}"}}"#, expr_src, escaped))
                        }
                        Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"{}"}}"#, e)),
                    }
                }
                Err(e) => ("400 Bad Request", format!(r#"{{"ok":false,"error":"parse: {}"}}"#, e)),
            }
        }

        _ => ("404 Not Found", r#"{"error":"not found"}"#.into()),
    }
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    // Simple extractor: finds "key":"value" pattern
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim();
    if rest.starts_with('"') {
        let inner = &rest[1..];
        let end = inner.find('"')?;
        Some(inner[..end].to_string())
    } else {
        None
    }
}
