import gym
import numpy as np
import streamlit as st
from streamlit import sidebar as sbar
from ansi2html import Ansi2HTMLConverter
import streamlit.components.v1 as components

import src.envs


def init():
    st.set_page_config(page_title="Interact", page_icon=":snake:", layout="wide")
    if "env" not in st.session_state:
        st.session_state.env = None
        st.session_state.tokenizer = None
        st.session_state.state = None
        st.session_state.references = []
        st.session_state.render_outputs = []
        st.session_state.reward_config = None


def reset():
    env = st.session_state.env
    st.session_state.state = env.reset()
    st.session_state.references = [
        st.session_state.tokenizer.tokens2text(ref_tokens)
        for ref_tokens in env.reference_tokens_list
    ]
    ansi_out = env.render()[0]
    st.session_state.render_outputs = [ansi_out]


def select_env():
    info = """
    Load and reset GEC environment
    """
    sbar.header("Environment Setup")
    sbar.markdown(info)
    env_id = sbar.text_input("Environment ID", value="lang8_gec-v0", key="env_id")
    cols = sbar.columns(2)
    load_btn = cols[0].button("Load", key="load_btn")
    reset_btn = cols[1].button("Reset", key="reset_btn", on_click=reset)
    if load_btn:
        if st.session_state.env is not None:
            st.session_state.env.close()
        st.session_state.env = env = gym.make(env_id)
        st.session_state.tokenizer = env.tokenizer
        st.session_state.reward_config = env.reward_config
        reset()


def display_env():
    st.markdown("### Environment Info")
    if st.session_state.env is not None:
        env_id = st.session_state.env.spec.id
        st.markdown(f"**Environment:** `{env_id}`")
    cols = st.columns(2)
    if st.session_state.reward_config is not None:
        with cols[0]:
            st.markdown("##### Reward Config:")
            st.json(st.session_state.reward_config, expanded=False)
    if st.session_state.state is not None:
        st.markdown("##### State Info:")
        st.markdown("###### Current State:")
        current_state = st.session_state.tokenizer.tokens2text(st.session_state.state)
        st.markdown(f"- {current_state}")
        st.markdown("###### References")
        for ref in st.session_state.references:
            st.markdown(f"- {ref}")


def select_action():
    info = """
    ### Apply labels to the tokens:
    - Select the tokens to apply labels
    - Select labels for selected tokens
    - Press `Apply Action` button to apply labels on the selected tokens
    """
    sbar.header("Action Interface")
    sbar.markdown(info)
    if st.session_state.state is not None:
        tokens = st.session_state.state
        labels = st.session_state.env.labels
        tok_options = range(len(tokens))
        act_options = range(len(labels))
        selected_tok_indexes = sbar.multiselect(
                "Selected Tokens",
                options=tok_options,
                format_func=tokens.__getitem__,
                key="token_idx",
                help="Select tokens to apply actions"
        )
        if selected_tok_indexes:
            sbar.markdown("Actions:")
        actions = np.zeros(len(tokens), dtype="uint32")
        for i in range(len(tokens)):
            if i in selected_tok_indexes:
                actions[i] = sbar.selectbox(
                        f"[{i}] {tokens[i]}",
                        options=act_options,
                        format_func=labels.__getitem__,
                        key=f"action_{i}",
                        help="Select action for this token"
                )
        apply_btn = sbar.button("Apply Action", key="apply_btn")
        if apply_btn:
            s, r, d, info = st.session_state.env.step(actions)
            st.session_state.state = s
            for ansi_out in st.session_state.env.render():
                st.session_state.render_outputs.append(ansi_out)


def render():
    if st.session_state.render_outputs:
        st.markdown("### Render Outputs:")
        render_out = "\n".join(st.session_state.render_outputs)
        html_out = Ansi2HTMLConverter().convert(render_out)
        components.html(html_out, height=600, scrolling=True)


def display_sidebar():
    select_env()
    sbar.markdown("---")
    select_action()


def display_body():
    display_env()
    st.markdown("---")
    render()


def main():
    init()
    display_sidebar()
    display_body()


if __name__ == "__main__":
    main()
