"""Local Gradio interface for CerS-M image/video inference with optional RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gradio as gr

DEMO_DIR = Path(__file__).resolve().parent
CERS_M_SRC = DEMO_DIR.parent / "src"
if str(CERS_M_SRC) not in sys.path:
    sys.path.insert(0, str(CERS_M_SRC))

from utils import disable_torch_init, get_model_name_from_path, load_pretrained_model

from rag_pipeline import RAGRefinePipeline
from retriever import MilvusHybridRetriever

VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".webm",
    ".wmv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local CerS-M image/video inference with optional RAG."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    parser.add_argument("--rag-db")
    parser.add_argument("--rag-collection", default="cervix")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--rag-device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a Gradio public link. Do not use with restricted clinical data.",
    )
    return parser.parse_args()


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def file_path(file_item: Any) -> str:
    if isinstance(file_item, str):
        return file_item
    if isinstance(file_item, dict):
        return str(file_item.get("path") or file_item.get("name"))
    return str(getattr(file_item, "path", None) or getattr(file_item, "name", file_item))


def history_to_conversation(history: list[Any] | None) -> list[dict[str, Any]]:
    """Convert either tuple-style or message-style Gradio history."""
    conversation: list[dict[str, Any]] = []
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            content = item.get("content")
            if isinstance(content, str):
                conversation.append(
                    {
                        "role": item["role"],
                        "content": [{"type": "text", "text": content}],
                    }
                )
            continue

        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        user_turn, assistant_turn = item
        if isinstance(user_turn, str):
            conversation.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_turn}],
                }
            )
        if isinstance(assistant_turn, str):
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_turn}],
                }
            )
    return conversation


def create_demo(args: argparse.Namespace) -> gr.Blocks:
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=args.device,
        device=args.device,
        use_flash_attn=args.use_flash_attn,
    )

    retriever = None
    if args.rag_db:
        retriever = MilvusHybridRetriever(
            db_path=args.rag_db,
            collection_name=args.rag_collection,
            embedding_model=args.embedding_model,
            device=args.rag_device,
        )

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
    }
    pipeline = RAGRefinePipeline(
        model=model,
        processor=processor,
        retriever=retriever,
        generation_args=generation_args,
        device=args.device,
    )

    def respond(message: dict[str, Any], history: list[Any]):
        images: list[str] = []
        videos: list[str] = []
        for uploaded_file in message.get("files") or []:
            path = file_path(uploaded_file)
            (videos if is_video_file(path) else images).append(path)

        question = str(message.get("text") or "")
        if not question and not images and not videos:
            yield "Please enter a question or upload an image/video."
            return

        try:
            final_answer, initial_answer, retrieved = pipeline.run(
                question=question,
                images=images,
                videos=videos,
                conversation=history_to_conversation(history),
                top_k=args.top_k,
            )
        except Exception as exc:
            yield f"Inference failed: {type(exc).__name__}: {exc}"
            return

        if retrieved:
            yield (
                f"{final_answer}\n\n"
                f"Retrieval note: refined with {len(retrieved)} reference chunk(s)."
            )
        else:
            yield initial_answer

    textbox = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", "video"],
        placeholder="Enter a question or upload a pathology image/video...",
        show_label=False,
    )
    chatbot = gr.Chatbot(scale=2)
    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown(
            "# CerS-M image/video demo\n"
            "Research use only; this interface is not a clinical diagnostic service."
        )
        gr.ChatInterface(
            fn=respond,
            multimodal=True,
            textbox=textbox,
            chatbot=chatbot,
        )
    return demo


def main() -> None:
    args = parse_args()
    demo = create_demo(args)
    demo.queue(api_open=False)
    demo.launch(
        show_api=False,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()

