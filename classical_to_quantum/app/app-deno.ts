// @ts-ignore
import {serve} from "https://deno.land/std@0.114.0/http/server.ts";

// @ts-ignore
const handleRequest = async (request: Request): Promise<Response> => {
    const url = new URL(request.url);

    // Redirect all requests to the Flask server
    const backendUrl = `http://127.0.0.1:5001${url.pathname}${url.search}`;

    // Proxy the request to the backend Flask server
    // Return the response from the Flask server
    return await fetch(backendUrl, {
        method: request.method,
        headers: request.headers,
        body: request.body,
    });
};

console.log("Deno server running on http://localhost:8000");
serve(handleRequest, { port: 8000 });
