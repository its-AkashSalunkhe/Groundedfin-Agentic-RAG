import gradio as gr
from agent import ask, app as agent_graph


def get_debug_info(thread_id):
    cfg = {"configurable": {"thread_id": thread_id}}
    state = agent_graph.get_state(cfg).values
    label = state.get("label", "?")
    grounded = state.get("grounded", None)
    kb_relevant = state.get("kb_relevant", None)

    parts = [f"**Classified as:** {label}"]
    if label in ("STATIC", "BOTH") and kb_relevant is not None:
        parts.append(f"**Found in RBI booklet:** {'yes' if kb_relevant else 'no — checked live news instead'}")
    if grounded is not None:
        parts.append(f"**Groundedness check:** {'passed' if grounded else 'needed a correction'}")
    return " &nbsp;|&nbsp; ".join(parts)


def chat_fn(message, history, user_id, request: gr.Request):
    thread_id = request.session_hash
    uid = (user_id or "").strip() or "guest"
    answer = ask(message, user_id=uid, thread_id=thread_id)
    debug = get_debug_info(thread_id)
    return answer, debug


with gr.Blocks(title="GroundedFin") as demo:
    gr.Markdown("# GroundedFin\nA trustworthy fraud & scam awareness agent for India, grounded in RBI's official guidance and live news.")

    user_id_box = gr.Textbox(label="Your name (optional — lets GroundedFin remember you across sessions)", value="")
    debug_display = gr.Markdown(value="*Ask a question to see how GroundedFin routed it.*")

    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[user_id_box],
        additional_outputs=[debug_display],
        examples=[
            ["What is vishing?", ""],
            ["Any recent UPI fraud cases in India?", ""],
            ["What should I do after falling for a fraud?", ""],
            ["What's a good recipe for pasta?", ""],
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)