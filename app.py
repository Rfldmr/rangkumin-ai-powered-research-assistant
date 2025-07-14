import os
import tempfile
import streamlit as st
import time
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

DEEPSEEK_API_KEY = st.secrets["my_api_key"]
DEEPSEEK_API_BASE = "https://openrouter.ai/api/v1"

def initialize_llm():
    """Initialize the DeepSeek v3 LLM through OpenRouter"""
    return ChatOpenAI(
        model_name="deepseek/deepseek-chat:free",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_API_BASE,
        temperature=0.3,
        max_tokens=4000
    )

def load_and_split_pdf(pdf_file):
    """Load and split the PDF into chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(pages)
        return splits
    finally:
        os.unlink(tmp_file_path)

def generate_summary(document_text):
    """Generate a structured summary of the journal paper"""
    llm = initialize_llm()
    
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Anda adalah asisten penelitian akademik. Ekstrak informasi berikut dari jurnal dalam format poin-poin tanpa kalimat pengantar atau penjelasan tambahan:

- **Judul**: [Judul jurnal]
(beri jarak)
- **Author(s)**: [Daftar penulis]
(beri jarak)
- **Abstract**: [Ringkasan konten jurnal]
(beri jarak)
- **Keywords**: [Kata kunci penting]
(beri jarak)
- **Metode Penelitian**: [Metodologi yang digunakan]
(beri jarak)
- **Publisher**: [Penerbit atau nama jurnal] atau 'Tidak Diketahui'
(beri jarak)
- **Tanggal Publish**: [Tanggal publikasi] atau 'Tidak Diketahui'
(beri jarak)
- **Kesimpulan**: [Kesimpulan utama penelitian]
(beri jarak)
- **DOI**: [Link atau nomor DOI] atau 'Tidak Diketahui'

Aturan ketat:
1. Hanya output informasi dalam format poin di atas
2. Tidak ada kalimat pengantar
3. Tidak ada catatan atau penjelasan tambahan
4. Jika informasi tidak ada, cukup tulis 'Tidak Diketahui'
5. Gunakan bahasa Indonesia secara konsisten
6. Berikan jarak setidaknya 1 baris disetiap poinnya, misal judul lalu beri baris, author lalu beri baris, abstrak lalu beri baris"""),
    ("user", "{document_text}")
])
    
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"document_text": document_text})

def chat_with_pdf(question, document_text):
    """Generate answer based on the PDF content"""
    llm = initialize_llm()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Kamu adalah personifikasi dari jurnal akademik ini. 
        Jawablah pertanyaan seolah-olah kamu adalah jurnal yang sedang berbicara (gunakan sudut pandang pertama/persona pertama).
        Gunakan kata ganti "aku" untuk merujuk pada jurnal ini.
        Berikan jawaban hanya berdasarkan informasi yang ada dalam konten jurnal.
        Jika pertanyaan tidak relevan dengan isi jurnal, katakan bahwa kamu tidak bisa menjawab karena itu di luar konteks dirimu.
        
        Contoh:
        User: Kapan kamu dipublish?
        Jawab: Aku dipublikasikan pada tanggal 20 Desember 2023
        
        User: Siapa penulismu?
        Jawab: Penulisku adalah [nama penulis] dan tim penelitian dari [institusi]
        
        Isi jurnal:
        {document_text}"""),
        ("user", "{question}")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"question": question, "document_text": document_text})

def generate_citations(document_text):
    """Generate citations in various styles"""
    llm = initialize_llm()
    
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert academic research assistant. Generate citations in multiple styles strictly following these rules:

1. Output ONLY the citation formats below, nothing else
2. No introductory sentences or explanations
3. Use this exact format for each style:

**APA Style**
<br>
[APA citation]

**MLA Style**
<br>
[MLA citation]

**Harvard Style**
<br>
[Harvard citation]

**IEEE Style**
<br>
[IEEE citation]

**Chicago Style**
<br>
[Chicago citation]

**Vancouver Style**
<br>
[Vancouver citation]

**AMA Style**
<br>
[AMA citation]

4. Include all available elements: authors, title, journal, year, volume, issue, pages, DOI
5. For missing information, use [assumed information] and keep it minimal
6. Do not add any other text outside the citation formats"""),
    ("user", "{document_text}")
])
    
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"document_text": document_text})

def find_related_journals(document_text):
    """Find related journals using arXiv API"""
    llm = initialize_llm()
    
    keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", """Anda adalah asisten penelitian yang ahli dalam menganalisis judul jurnal akademik. Ekstrak 3-5 frasa kunci terpenting dari judul jurnal yang akan digunakan untuk mencari paper sejenis. 

Aturan:
1. Ambil konsep inti, metode, teknologi, dan objek penelitian
2. Prioritaskan istilah teknis/spesifik
3. Gabungkan kata yang harus berdampingan (contoh: "regresi linear")
4. Abaikan kata umum seperti "analisis", "studi", "penggunaan" kecuali sangat relevan
5. Hasil HANYA berisi kata kunci dipisahkan koma, tanpa penjelasan
6. Terjemahkan hasil ke dalam bahasa Inggris

Contoh:
Judul: "Analisis Prediksi Harga Rumah Sesuai Spesifikasi Menggunakan Metode Regresi Linear Berganda Berbasis Shiny R"
Output: Prediksi harga rumah, regresi linear berganda, Shiny R

Judul: "Penerapan Metode Waterfall dalam Perencanaan Sistem Informasi Penjualan Buku berbasis Aplikasi Website (Studi Kasus: Penjual Buku Toko 21 Jombang)"
Output: Waterfall, Perencanaan, Sistem Informasi, Aplikasi, Website

Judul: "{document_text}"
Output: """),
    ("user", "{document_text}")
])
    
    keyword_chain = keyword_prompt | llm | StrOutputParser()
    keywords = keyword_chain.invoke({"document_text": document_text})
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=keywords,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for result in client.results(search):
            arxiv_keywords = [tag.term for tag in result.tags] if hasattr(result, 'tags') else []
        
            search_keywords = [kw.strip() for kw in keywords.split(',')] if keywords else []
            combined_keywords = list(set(arxiv_keywords + search_keywords))
            
            results.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "keywords": ", ".join(combined_keywords[:5]),
            "pdf_url": result.pdf_url,
            "doi": result.doi if result.doi else "Tidak tersedia",
        })
    except Exception as e:
        st.error(f"Error saat mencari di arXiv: {str(e)}")
        return []
    
    formatted_results = []
    for idx, paper in enumerate(results, 1):
        formatted = f"""
{idx}. **{paper['title']}**
- **Penulis**: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}  
- **Tanggal Publikasi**: {paper['published']}   
- **DOI**: {paper['doi']} 
- **Keyword**: {paper.get('keywords', 'Tidak tersedia')} 

**Ringkasan**:  
{paper['summary'][:300]}...  

[📄 Unduh PDF]({paper['pdf_url']}) | [🔗 Lihat Detail](https://arxiv.org/abs/{paper['pdf_url'].split('/')[-1].replace('.pdf','')})
"""
        formatted_results.append(formatted)
    
    return formatted_results

def main():
    st.set_page_config(page_title="AI-Powered Research Assistant", page_icon="📚", layout="wide")
    
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "summary"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None
    if 'full_text' not in st.session_state:
        st.session_state.full_text = ""
    if 'citations' not in st.session_state:
        st.session_state.citations = None
    if 'related_papers' not in st.session_state:
        st.session_state.related_papers = None
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.title("📚 RangkumIN AI")
        
        if st.session_state.summary is None:
            st.markdown("### AI-Powered Research Assistant")
            st.markdown("")
            st.markdown("""
            RangkumIN adalah research assistant berbasis AI yang akan membantu kamu mempermudah proses penelitian terhadap jurnal-jurnal terdahulu yang ingin kamu gunakan.
            """)
            st.markdown("")
            st.markdown("""
            **Cara Penggunaan:**
            - 📤 Upload jurnal yang ingin diekstraksi.
            - 📋 Pastikan jurnal berada dalam format PDF.
            - ⚙️ Pilih opsi menu yang kamu butuhkan. 
            """)
        else:
            st.markdown("### AI-Powered Research Assistant")
            st.markdown("Apa yang bisa aku bantu buat kamu hari ini?")
            
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                if st.button("📄 Hasil Ringkasan", use_container_width=True):
                    st.session_state.current_mode = "summary"
            with col1_2:
                if st.button("💬 Ajak Ngobrol Jurnal", use_container_width=True):
                    st.session_state.current_mode = "chat"
            
            if st.button("📝 Buat Sitasi Otomatis", use_container_width=True):
                st.session_state.current_mode = "citation"
            if st.button("🔍 Jelajahi Jurnal Terkait", use_container_width=True):
                st.session_state.current_mode = "related_journals"
                st.session_state.related_papers = None
        
        st.markdown("")
        uploaded_file = st.file_uploader("Pilih file PDF", type=["pdf"], key="uploader")
        
        st.divider()
        st.caption("© 2025 Rafli Damara")
        
    with col2:
        if uploaded_file is not None and (st.session_state.summary is None or 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file.name):
            st.subheader("📄 Hasil Ringkasan")
            st.markdown("")
            with st.spinner("Sabar yaaa, jurnalnya lagi aku baca... 🤓"):
                try:
                    st.session_state.uploaded_file = uploaded_file.name
                    splits = load_and_split_pdf(uploaded_file)
                    st.session_state.full_text = "\n\n".join([doc.page_content for doc in splits])
                    st.session_state.summary = generate_summary(st.session_state.full_text)
                    st.session_state.current_mode = "summary"
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
                    st.error("Pastikan file PDF berisi teks yang dapat dibaca (bukan scan gambar).")
        
        if st.session_state.current_mode == "summary":
            st.subheader("📄 Hasil Ringkasan")
            if st.session_state.summary:
                st.markdown("")
                st.markdown(st.session_state.summary)
            else:
                st.markdown("")
                st.info("Silakan unggah file PDF di kolom sebelah kiri untuk memulai")
        
        elif st.session_state.current_mode == "chat":
            st.subheader("💬 Ayo Ajak Ngobrol Jurnal Kamu!")
            st.markdown("")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            if len(st.session_state.chat_history) == 0:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "Kamu mau tau apa tentang aku?"
                })
            
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            if prompt := st.chat_input("Ajukan pertanyaan tentang jurnal ini..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        with st.spinner("Bentar mikir dulu yaa..."):
                            time.sleep(1)
                            response = chat_with_pdf(prompt, st.session_state.full_text)
                            message_placeholder.markdown(response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
        
        elif st.session_state.current_mode == "citation":
            st.subheader("📝 Sitasi Jurnal")
            st.markdown("")
            
            if st.session_state.citations is None:
                with st.spinner("Sabar yaaa, sitasi jurnal ini lagi aku susun... 😁"):
                    st.session_state.citations = generate_citations(st.session_state.full_text)
                    st.rerun()
            else:
                st.markdown(st.session_state.citations)
        
        elif st.session_state.current_mode == "related_journals":       
            st.subheader("🔍 Jurnal Terkait")
            
            if st.session_state.related_papers is None:
                with st.spinner("Sabar yaaa, dicari dulu... 🧐"):
                    try:
                        st.session_state.related_papers = find_related_journals(st.session_state.full_text)
                        if not st.session_state.related_papers:
                            st.warning("Tidak ditemukan jurnal terkait. Coba dengan kata kunci yang lebih spesifik.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal mencari jurnal terkait: {str(e)}")
                        st.session_state.related_papers = []
            
            if st.session_state.related_papers:
                st.markdown("Berikut adalah jurnal terkait yang berhasil ditemukan:")
                st.markdown("---")
                for paper in st.session_state.related_papers:
                    st.markdown(paper, unsafe_allow_html=True)
                    st.markdown("---")
                
                if st.button("Coba Cari Lagi", type="primary"):
                    st.session_state.related_papers = None
                    st.rerun()
            else:
                st.info("Tidak ada hasil yang ditemukan. Coba unggah jurnal dengan konten yang lebih spesifik.")

if __name__ == "__main__":
    main()
