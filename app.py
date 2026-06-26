import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide')

if "df" not in st.session_state:
    st.session_state.df = None

np.random.seed(42)

n = 30
x = np.linspace(-5,5.5,n)
x2 = np.random.normal(0, 2.5, n)
x3 = np.random.randint(1, 8, n)
t = np.linspace(0,2*np.pi,128)

@st.dialog("Penjelasan Mengenai Data Melingkar")
def penjelas_circle():
    st.markdown("""
    Koefisien yang diestimasi pada metode **Least Square Circle Fitting** bukanlah
    pusat dan jari-jari lingkaran secara langsung.

    Persamaan lingkaran terlebih dahulu diubah menjadi bentuk linier sehingga parameter
    yang diestimasi adalah **D**, **E**, dan **F**.
    """)

    st.latex(r"(x-h)^2+(y-k)^2=r^2")

    st.markdown("Dikembangkan menjadi")

    st.latex(r"x^2+y^2-2hx-2ky+h^2+k^2-r^2=0")

    st.markdown("Kemudian ditulis sebagai")

    st.latex(r"x^2+y^2+Dx+Ey+F=0")

    st.markdown("Sehingga hubungan parameter adalah")

    st.latex(r"D=-2h \qquad atau \qquad \beta_0")
    st.latex(r"E=-2k \qquad atau \qquad \beta_1")
    st.latex(r"F=h^2+k^2-r^2 \qquad atau \qquad \beta_2")

    st.divider()

    st.markdown("Untuk dataset simulasi ini:")

    st.latex(r"x^2+y^2=25")

    st.markdown("atau")

    st.latex(r"x^2+y^2-25=0")

    st.markdown("Sehingga koefisien sebenarnya adalah")

    st.latex(r"D=0,\qquad E=0,\qquad F=-25")

    st.info(
        "Nilai hasil estimasi tidak akan sama persis karena setiap titik "
        "ditambahkan noise Gaussian."
    )

def f_num(x):
    return f"{float(x):.4g}"

def _matrix_visual_story(data):
    if data == "Linear":
        A = r"\begin{bmatrix}1&x_1\\1&x_2\\\vdots&\vdots\\1&x_n\end{bmatrix}"
        beta = r"\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}"
        b = r"\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}"
        product = rf"\underbrace{{{A}}}_{{A}}\;\underbrace{{{beta}}}_{{\hat{{\beta}}}}=\underbrace{{{b}}}_{{b}}"

        return [
            {"title": "1. Model awal", "kind": "single", "latex": r"y_i=\beta_0+\beta_1x_i+e_i"},
            {"title": "2. Bentuk matriks", "kind": "single", "latex": r"\mathbf{b}=A\hat{\beta}+e"},
            {
                "title": "3. Susunan matriks",
                "kind": "triplet",
                "A": rf"A={A}",
                "beta": rf"\hat{{\beta}}={beta}",
                "b": rf"b={b}",
                "prod": product,
            },
            {"title": "4. Fungsi error", "kind": "single", "latex": r"E(\beta)=(b-A\beta)^T(b-A\beta)"},
            {"title": "5. Persamaan normal", "kind": "single", "latex": r"\hat{\beta}=(A^TA)^{-1}A^Tb"},
        ]

    if data == "Multi Variabel":
        A = r"\begin{bmatrix}1&x_{11}&x_{12}&x_{13}\\1&x_{21}&x_{22}&x_{23}\\\vdots&\vdots&\vdots&\vdots\\1&x_{n1}&x_{n2}&x_{n3}\end{bmatrix}"
        beta = r"\begin{bmatrix}\beta_0\\\beta_1\\\beta_2\\\beta_3\end{bmatrix}"
        b = r"\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}"
        product = rf"\underbrace{{{A}}}_{{A}}\;\underbrace{{{beta}}}_{{\hat{{\beta}}}}=\underbrace{{{b}}}_{{b}}"

        return [
            {"title": "1. Model awal", "kind": "single", "latex": r"y_i=\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\beta_3x_{i3}+e_i"},
            {"title": "2. Bentuk matriks", "kind": "single", "latex": r"\mathbf{b}=A\hat{\beta}+e"},
            {
                "title": "3. Susunan matriks",
                "kind": "triplet",
                "A": rf"A={A}",
                "beta": rf"\hat{{\beta}}={beta}",
                "b": rf"b={b}",
                "prod": product,
            },
            {"title": "4. Fungsi error", "kind": "single", "latex": r"E(\beta)=(b-A\beta)^T(b-A\beta)"},
            {"title": "5. Persamaan normal", "kind": "single", "latex": r"\hat{\beta}=(A^TA)^{-1}A^Tb"},
        ]

    if data == "Polinomial (kuadrat)":
        A = r"""
        \begin{bmatrix}
        1 & x_i & x_i^2\\
        1 & x_i & x_i^2\\
        \vdots & \vdots & \vdots\\
        1 & x_i & x_i^2
        \end{bmatrix}
        """
        beta = r"\begin{bmatrix}\beta_0\\\beta_1\\\beta_2\end{bmatrix}"
        b = r"\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}"
        product = rf"\underbrace{{{A}}}_{{A}}\;\underbrace{{{beta}}}_{{\hat{{\beta}}}}=\underbrace{{{b}}}_{{b}}"

        return [
            {"title": "1. Model awal", "kind": "single", "latex": r"y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+e_i"},
            {"title": "2. Bentuk matriks", "kind": "single", "latex": r"\mathbf{b}=A\hat{\beta}+e"},
            {
                "title": "3. Susunan matriks",
                "kind": "triplet",
                "A": rf"A={A}",
                "beta": rf"\hat{{\beta}}={beta}",
                "b": rf"b={b}",
                "prod": product,
            },
            {"title": "4. Fungsi error", "kind": "single", "latex": r"E(\beta)=(b-A\beta)^T(b-A\beta)"},
            {"title": "5. Persamaan normal", "kind": "single", "latex": r"\hat{\beta}=(A^TA)^{-1}A^Tb"},
        ]

    if data == "Polinomial (kubik)":
        A = r"""
        \begin{bmatrix}
        1 & x_i & x_i^2 & x_i^3\\
        1 & x_i & x_i^2 & x_i^3\\
        \vdots & \vdots & \vdots & \vdots\\
        1 & x_i & x_i^2 & x_i^3
        \end{bmatrix}
        """
        beta = r"\begin{bmatrix}\beta_0\\\beta_1\\\beta_2\\\beta_3\end{bmatrix}"
        b = r"\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}"
        product = rf"\underbrace{{{A}}}_{{A}}\;\underbrace{{{beta}}}_{{\hat{{\beta}}}}=\underbrace{{{b}}}_{{b}}"

        return [
            {"title": "1. Model awal", "kind": "single", "latex": r"y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+\beta_3x_i^3+e_i"},
            {"title": "2. Bentuk matriks", "kind": "single", "latex": r"\mathbf{b}=A\hat{\beta}+e"},
            {
                "title": "3. Susunan matriks",
                "kind": "triplet",
                "A": rf"A={A}",
                "beta": rf"\hat{{\beta}}={beta}",
                "b": rf"b={b}",
                "prod": product,
            },
            {"title": "4. Fungsi error", "kind": "single", "latex": r"E(\beta)=(b-A\beta)^T(b-A\beta)"},
            {"title": "5. Persamaan normal", "kind": "single", "latex": r"\hat{\beta}=(A^TA)^{-1}A^Tb"},
        ]

    if data == "Circle":
        A = r"\begin{bmatrix}x_1&y_1&1\\x_2&y_2&1\\\vdots&\vdots&\vdots\\x_n&y_n&1\end{bmatrix}"
        beta = r"\begin{bmatrix}D\\E\\F\end{bmatrix}"
        b = r"\begin{bmatrix}-(x_1^2+y_1^2)\\-(x_2^2+y_2^2)\\\vdots\\-(x_n^2+y_n^2)\end{bmatrix}"
        product = rf"\underbrace{{{A}}}_{{A}}\;\underbrace{{{beta}}}_{{\hat{{\beta}}}}=\underbrace{{{b}}}_{{b}}"

        return [
            {"title": "1. Model awal", "kind": "single", "latex": r"x_i^2+y_i^2+Dx_i+Ey_i+F=0"},
            {"title": "2. Bentuk linear", "kind": "single", "latex": r"Dx_i+Ey_i+F=-(x_i^2+y_i^2)"},
            {
                "title": "3. Susunan matriks",
                "kind": "triplet",
                "A": rf"A={A}",
                "beta": rf"\hat{{\beta}}={beta}",
                "b": rf"b={b}",
                "prod": product,
            },
            {"title": "4. Fungsi error", "kind": "single", "latex": r"E(\beta)=(b-A\beta)^T(b-A\beta)"},
            {"title": "5. Persamaan normal", "kind": "single", "latex": r"\hat{\beta}=(A^TA)^{-1}A^Tb"},
        ]

    A = r"\begin{bmatrix}1&x_1\\1&x_2\\\vdots&\vdots\\1&x_n\end{bmatrix}"
    beta = r"\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}"
    b = r"\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}"
    product = rf"\underbrace{{{A}}}_{{A}}\;\underbrace{{{beta}}}_{{\hat{{\beta}}}}=\underbrace{{{b}}}_{{b}}"

    return [
        {"title": "1. Model awal", "kind": "single", "latex": r"y_i=\beta_0+\beta_1x_i+e_i"},
        {"title": "2. Bentuk matriks", "kind": "single", "latex": r"\mathbf{b}=A\hat{\beta}+e"},
        {
            "title": "3. Susunan matriks",
            "kind": "triplet",
            "A": rf"A={A}",
            "beta": rf"\hat{{\beta}}={beta}",
            "b": rf"b={b}",
            "prod": product,
        },
        {"title": "4. Fungsi error", "kind": "single", "latex": r"E(\beta)=(b-A\beta)^T(b-A\beta)"},
        {"title": "5. Persamaan normal", "kind": "single", "latex": r"\hat{\beta}=(A^TA)^{-1}A^Tb"},
    ]

def least_square(A, b):
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1, 1)

    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    fitting = A @ beta
    residual = b - fitting

    sse = np.sum(residual**2)
    mse = np.mean(residual**2)
    rmse = np.sqrt(mse)
    r2 = 1 - sse / np.sum((b - b.mean())**2)

    return {
        "beta": beta.ravel(),
        "fitting": fitting.ravel(),
        "residual": residual.ravel(),
        "SSE": float(sse),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

def _build_latex(beta, data):
    beta = np.asarray(beta, dtype=float)

    if data == "Circle":
        latex = r"x^2 + y^2"
        for coef, var in zip(beta, ["x", "y", ""]):
            tanda = "+" if coef >= 0 else "-"
            if var:
                latex += rf" {tanda} {f_num(abs(coef))}{var}"
            else:
                latex += rf" {tanda} {f_num(abs(coef))}"
        return latex + r" = 0"

    if data == "Linear":
        tanda = "+" if beta[1] >= 0 else "-"
        return rf"\hat{{y}} = {f_num(beta[0])} {tanda} {f_num(abs(beta[1]))}x"

    is_poly = data in ["Polinomial (kuadrat)", "Polinomial (kubik)"]
    latex = rf"\hat{{y}} = {f_num(beta[0])}"

    for i, coef in enumerate(beta[1:], start=1):
        tanda = "+" if coef >= 0 else "-"
        if is_poly:
            variabel = "x" if i == 1 else rf"x^{{{i}}}"
        else:
            variabel = rf"x_{{{i}}}"

        latex += rf" {tanda} {f_num(abs(coef))}{variabel}"

    return latex

Dataset = {
    "Linear": {
        "data": pd.DataFrame({
            "x": x,
            "y": 2.3*x - 1 + np.random.normal(0, 1, n)
        }),
        "latex": (
            r"y=-1+2.3x+\varepsilon",
            r"\qquad \varepsilon\sim\mathcal{N}(0,1)"
        ),
        "beta": [-1, 2.3]
    },

    "Multi Variabel": {
        "data": pd.DataFrame({
            "x1": x,
            "x2": x2,
            "x3": x3,
            "y": 2.5*x - 1.8*x2 + 3.2*x3 + 1.2 + np.random.normal(0, 1, n)
        }),
        "latex": (
            r"y=1.2+2.5x_1-1.8x_2+3.2x_3+\varepsilon",
            r"\qquad \varepsilon\sim\mathcal{N}(0,1)"
        ),
        "beta": [1.2, 2.5, -1.8, 3.2]
    },

    "Polinomial (kuadrat)": {
        "data": pd.DataFrame({
            "x": x,
            "x^2": x**2,
            "y": -0.2*x**2 - 0.25*x + 5 + np.random.normal(0, 0.5, n)
        }),
        "latex": (
            r"y=5-0.25x-0.2x^2+\varepsilon",
            r"\qquad \varepsilon\sim\mathcal{N}(0,0.5)"
        ),
        "beta": [5, -0.25, -0.2]
    },

    "Polinomial (kubik)": {
        "data": pd.DataFrame({
            "x": x,
            "x^2": x**2,
            "x^3": x**3,
            "y": 0.25*x**3 + 0.25*x**2 - 5*x + np.random.normal(0, 2.5, n)
        }),
        "latex": (
            r"y=-5x+0.25x^2+0.25x^3+\varepsilon",
            r"\qquad \varepsilon\sim\mathcal{N}(0,2.5)"
        ),
        "beta": [0, -5, 0.25, 0.25]
    },

    "Circle": {
        "data": pd.DataFrame({
            "x": 5*np.cos(t) + np.random.normal(0, 0.5, len(t)),
            "y": 5*np.sin(t) + np.random.normal(0, 0.5, len(t))
        }),
        "latex": (
            r"x^2+y^2=25",
            r"\qquad \varepsilon_x,\varepsilon_y\sim\mathcal{N}(0,0.5)"
        ),
        "beta": [0, 0, -25]
    }
}

st.title("Metode Least Square")
st.caption("Made with ❤️ by **Wira Triono**")
st.markdown("""
        Disaat anda ingin melakukan regresi, baik sederhana maupun berganda menggunakan metode *least square*, 
        pasti anda sering melihat rumus-rumus seperti ini:
        """)

col1, col2, col3 = st.columns(3, border=True, vertical_alignment='center')

col1.latex(r'\hat{b}_0 = \frac{\sum X^2 \sum Y - \sum X \sum XY}{n \sum X^2 - (\sum X)^2}')
col1.latex(r'\hat{b}_1 = \frac{n \sum XY - \sum X \sum Y}{n \sum X^2 - (\sum X)^2}')

col2.latex(r'\beta = (X^T X)^{-1} X^T y')

col3.latex(r'b = \frac{n \sum XY - \sum X \sum Y}{n \sum X^2 - (\sum X)^2}')
col3.latex(r'a = \frac{\sum Y}{n} - \frac{b \sum X}{n} \text{ atau } a=\bar{Y} - b\bar{X}')


st.markdown("""
    Tapi pernahkah anda bertanya-tanya\n
    - “*kok bisa rumusnya seperti itu?*”<br>
    - “*dari ketiga rumus tersebut, mana yang benar?*”<br>
    - “*Mengapa ada bentuk pecahan, ada penjumlahan, ada transpose, bahkan invers matriks?*”<br>

    Menariknya, ketiga rumus yang ditampilkan sebenarnya sama-sama :green-badge[**benar**]. Perbedaannya hanya terletak pada cara penulisannya. 
    Rumus dalam bentuk matriks merupakan bentuk model linier pada umumnya dan dapat digunakan untuk regresi dengan banyak variabel, 
    sedangkan rumus yang berbentuk pecahan merupakan hasil penyederhanaan dari bentuk matriks tersebut untuk kasus regresi linier sederhana satu variabel [[1]](https://doi.org/10.51329/mehdiophthal1506).

    Lalu bagaimana proses penyederhanaan tersebut terjadi? Mengapa dari sebuah persamaan matriks yang terlihat cukup rumit dapat muncul rumus yang jauh lebih sederhana?
    """, unsafe_allow_html=True, text_alignment='justify')

tab1, tab2, tab3, tab4 = st.tabs(['📐 Penurunan Rumus', '✨ Penyederhanaan Rumus', '📚 Referensi', '🚀 Simulasi'])

with tab1:
    st.markdown(r"""
            Jika diberikan sebuah data sebanyak $n$ dengan $m$ variabel independen dan 1 variabel dependen,
            maka secara umum model linier yang terbentuk adalah sebagai berikut [[2]](https://online.stat.psu.edu/stat462/node/132/).
            """)
    st.latex(r'Y = X \beta + \epsilon \qquad dengan \qquad \epsilon \sim N(0, \sigma^2I)')

    col1, col2 = st.columns(2)
    col1.markdown(r'''
            Dimana :
            - $Y$ = variabel dependen
            - $X$ = variabel independen
            - $\beta$ = koefisien model
            ''')


    st.markdown(r"""
            Pertanyaan berikutnya adalah, **bagaimana cara mengukur kualitas suatu model?** <br>
            Model yang terbaik adalah model yang menghasilkan error paling rendah. 
            Error merupakan selisih antara nilai aktual dengan nilai hasil prediksi model, 
            sehingga error menunjukkan seberapa jauh hasil prediksi model menyimpang dari data sebenarnya [[2]](https://online.stat.psu.edu/stat462/node/132/). 
            Untuk menghitung error, cara paling sederhana adalah menjumlahkan seluruh error $(\sum_{i=1}^{n} e_i)$.
            Namun dalam praktiknya, hasil prediksi model tidak selalu tepat berada pada nilai aktual. 
            Ada kalanya model menghasilkan prediksi yang lebih besar dari nilai aktual, dan ada pula yang lebih kecil dari nilai aktual. 
            Akibatnya, error dapat bernilai positif maupun negatif. Maka dari itu, jika nilai error dijumlahkan secara langsung, 
            sebagian error positif dan negatif dapat saling menghilangkan [[3]](https://doi.org/10.1016/j.apenergy.2024.122753).
            sehingga model yang sebenarnya kurang baik dapat terlihat memiliki total error yang kecil.
            Untuk mengatasi masalah tersebut, setiap error dikuadratkan terlebih dahulu sehingga 
            seluruh nilai error menjadi positif, ini lah yang disebut dengan :green-badge[*Sum of Squared Errors*] (SSE) [[4]](https://www.wiley-vch.de/en/areas-interest/mathematics-statistics/introduction-to-linear-regression-analysis-978-1-119-57872-7).
            Karena persamaan umum menggunakan bentuk matriks, maka SSE perlu ditransformasikan dari bentuk skalar ke bentuk vektor.
            """, text_alignment='justify', unsafe_allow_html=True)

    st.latex(r'\sum_{i=1}^{n} e_i^2 = e^Te')
    with st.expander("penjelasan"):
        st.markdown(r"Apabila suatu vektor $(n\times1)$  ditranspose dan kemudian dikalikan dengan dirinya sendiri, maka diperoleh:")

        st.latex(r'''
                e^T e =
                \begin{bmatrix}
                e_1 & e_2 & \cdots & e_n
                \end{bmatrix}
                \begin{bmatrix}
                e_1\\
                e_2\\
                \vdots\\
                e_n
                \end{bmatrix}
                =
                e_1^2+e_2^2+\cdots+e_n^2 = \sum_{i=1}^{n} e_i^2
                ''')

        st.markdown(r"Dengan demikian, bentuk $e^T e$ sama dengan $\sum e_i^2$.")

    st.divider()
    st.markdown("langkah berikutnya adalah menyatakan error dalam bentuk persamaan model linier. Error tersebut dapat dituliskan sebagai:")
    st.latex(r'e = Y - X\beta')
    st.markdown(r"Dengan mensubstitusikan persamaan error $e = Y - X\beta$ ke dalam bentuk SSE ($e^T e$), sehingga diperoleh:")
    
    st.latex(r'e^T e = (Y - X\beta)^T (Y - X\beta)')
    st.markdown(r"""
                Karena tujuan metode *Least Square* adalah mencari nilai parameter $\beta$ yang menghasilkan SSE minimum [[5]](https://doi.org/10.1002/0471704091),
                maka bentuk $e^T e$ diubah ke dalam bentuk fungsi objektif yang akan diminimumkan. 
                Fungsi tersebut dinyatakan sebagai $E(\beta)$:
                """)
    st.latex(r'E(\beta) = (Y-X\beta)^T(Y-X\beta)')

    st.markdown(r"""
    Untuk memperoleh turunan dari fungsi objektif tersebut terhadap parameter $\beta$,
    bentuk matriks $E(\beta)$ terlebih dahulu dikembangkan menggunakan sifat transpose matriks
    """)

    st.markdown(r"""
                - :blue-badge[$(A-B)^T=A^T-B^T$]
                - :blue-badge[$AB^T=B^T A^T$]
                """)

    st.markdown("sehingga diperoleh:")
    st.latex(r'E(\beta) = (Y^T-\beta^TX^T)(Y-X\beta)')
    st.markdown("""
    Selanjutnya digunakan sifat distributif pada perkalian matriks,
    sama seperti proses mengalikan dua bentuk aljabar

    - :blue-badge[$(a-b)(c-d) = ac-ad-bc+bd$]
                
    Maka:
    """)

    st.latex(r'E(\beta) = Y^TY -Y^TX\beta -\beta^TX^TY +\beta^TX^TX\beta')
    st.markdown("""
    Perhatikan bahwa suku kedua dan suku ketiga sebenarnya bernilai sama.
    Hal ini karena keduanya merupakan skalar (berukuran $1\\times1$), sehingga:
    """)

    with st.expander('penjelasan'):
        col1, col2 = st.columns(2, border=True)
        col1.markdown(r"""
                Suku kedua
                    
                - $y^T = (1 \times n)$
                - $X_{(n \times m)} \beta_{(m \times 1)} = (n \times 1)$ 
                    
                maka :
                    
                $y^TX\beta = (1 \times n) (n \times 1)$
                    
                $y^TX\beta = (1 \times 1)$
                """)
        col2.markdown(r"""
                Suku ketiga 
                    
                - $\beta^T = (1 \times m)$
                - $X^T = (m \times n)$ 
                    
                maka :
                    
                $\beta^TX^Ty = (1 \times m) (m \times n) (n \times 1) $
                    
                $\beta^TX^Ty = (1 \times 1)$
                """)

    st.latex(r'(Y^TX\beta)^T = \beta^TX^TY')
    st.markdown("Karena transpose suatu skalar tidak mengubah nilainya, maka:")

    st.latex(r'Y^TX\beta = \beta^TX^TY')

    st.markdown("Akibatnya kedua suku tersebut dapat digabungkan menjadi:")
    st.latex(r'-Y^TX\beta -\beta^TX^TY=-2\beta^TX^TY')
    st.markdown("Sehingga bentuk akhir fungsi objektif tersebut menjadi:")

    st.latex(r'E(\beta) = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta')

    st.divider()
    st.markdown(r"""
                Langkah berikutnya adalah mencari nilai $\beta$ yang menghasilkan nilai persamaan $E(\beta)$ sekecil mungkin. 
                Dalam kalkulus, nilai minimum suatu fungsi dapat diterjadi ketika turunan pertama bernilai nol. 
                Karena nilai persamaan dipengaruhi oleh $\beta$, maka persamaan tersebut diturunkan terhadap $\beta$ sehingga sehingga diperoleh:
                """)

    st.latex(r'\frac{dE(\beta)}{d\beta}=0')

    st.markdown("Kemudian turunan terhadap $\\beta$ dihitung sehingga diperoleh:")
    with st.expander("penjelasan"):
        st.markdown(r"""
                Proses diferensiasi dilakukan pada setiap suku secara terpisah. 
                Tapi sebelum itu perhatikan beberapa aturan turunan matriks berikut:
                
                1. Jika $c$ adalah konstanta terhadap $\beta$, maka $\\ \frac{dc}{d\beta}=0$
                2. Jika $a$ adalah vektor konstan, maka $\\ \frac{d\beta^Ta}{d\beta}=a$
                3. Jika $A$ adalah matriks konstan dan simetris, maka $\\ \frac{d\beta^TA\beta}{d\beta}=2A\beta$
                """)

        st.markdown(r"""
                ---   
                ##### Suku pertama $\qquad(Y^TY)$
                """)
        st.markdown(r"""
                karena tidak mengandung parameter $\beta$, sehingga nilainya dianggap konstanta $(aturan\ 1)$.
                            
                $\frac{d(Y^TY)}{d\beta}=0$
                """)
        st.markdown(r"""
                ---
                ##### Suku kedua $\qquad(-2\beta^TX^TY)$
                """)
        st.markdown(r"""
                Karena $X^TY$ merupakan vektor konstan terhadap $\beta$, maka berlaku $(aturan\ 2)$ dengan $a=X^TY$.
                
                $\frac{d(-2\beta^TX^TY}{d\beta})=-2X^TY$
                """)

        st.markdown(r"""
                ---
                ##### Suku ketiga $\qquad(\beta^TX^TX\beta)$
                """)
        st.markdown(r"""
                Karena $X^TX$ adalah matriks konstan dan selalu simetris $((X^TX)^T=X^TX)$, maka berlaku $(aturan\ 3)$ dengan $A=X^TX$.
                
                $\frac{d(\beta^TX^TX\beta)}{d\beta})=-2X^TX\beta$
                """)
        st.divider()
        st.markdown("Dengan menggabungkan hasil turunan dari ketiga suku tersebut diperoleh:")
        with st.container(horizontal=True):
            st.latex(r'\frac{d E(\beta)}{d\beta}=0-2X^TY+2X^TX\beta')
            st.markdown("<br> atau dapat difaktorkan menjadi", text_alignment='center', unsafe_allow_html=True)
            st.latex(r'\frac{d E(\beta)}{d\beta}=2(-X^TY + X^TX\beta)')

    st.latex(r'\frac{d E(\beta)}{d\beta}=2(-X^TY + X^TX\beta)')

    st.markdown("Hasil turunan tersebut kemudian disamakan dengan nol:")
    st.latex(r'0 = 2(-X^TY+X^TX\beta)')

    st.markdown("Selanjutnya kedua ruas dibagi dengan 2 dan suku $-X^TY$ dipindahkan ke ruas kiri sehingga diperoleh:")
    st.latex(r'X^TX\beta=X^TY')

    st.markdown("""
                Untuk memperoleh nilai $\\beta$, kedua ruas dikalikan dengan invers dari $X^TX$.
                Dengan menggunakan sifat 
        
                - :blue-badge[$(X^TX)^{-1}(X^TX)=I$]
                
                diperoleh:
                """)

    st.latex(r'(X^TX)^{-1}(X^TX)\beta=(X^TX)^{-1}X^Ty')
    st.latex(r'I\beta=(X^TX)^{-1}X^Ty')
    st.markdown("""Dengan demikian diperoleh persamaan yang dikenal sebagai persamaan normal *(Normal Equation)* 
                [[4]](https://www.wiley-vch.de/en/areas-interest/mathematics-statistics/introduction-to-linear-regression-analysis-978-1-119-57872-7)[[5]](https://doi.org/10.1002/0471704091)""")
    st.markdown(r'#### :blue[$\hat\beta=(X^TX)^{-1}X^Ty$]', text_alignment='center')

    with st.expander("Penjelasan Rangkuman"):
        st.subheader("Aljabar")
        st.latex(r"""
                 \begin{array}{ll}
                 Y = X\beta + e & \cdots(1) \qquad \text{Persamaan umum regresi linier} \\[8pt]
                 e = Y - X\beta & \cdots(2) \qquad \text{Kurangi kedua ruas dengan } X\beta \\[8pt]
                 e^T e = (y - X\beta)^T (y - X\beta) & \cdots(3) \qquad \text{bentuk kuadrat dari pers.(2)} \\[8pt]
                 E(\beta) = (y-X\beta)^T(y-X\beta) & \cdots(4) \qquad \text{ubah $e^Te$ menjadi fungsi error $E(\beta)$} \\[8pt]
                 E(\beta) = (y^T-\beta^TX^T)(y-X\beta) & \cdots(5) \qquad \text{transpose matiks di dalam kurung} \\[8pt]
                 E(\beta) = y^Ty -y^TX\beta -\beta^TX^Ty +\beta^TX^TX\beta & \cdots(6) \qquad \text{perkalian aljabar} \\[8pt]
                 E(\beta) = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta & \cdots(7) \qquad (A-B)(A-B) = A^2 - 2AB + B^2 \\
                 \end{array}
                 """)
        st.subheader("Diferensial")
        st.latex(r"""
                 \begin{array}{ll}
                 \frac{dE(\beta)}{d\beta}=0 & \cdots(8) \qquad \text{definisi titik optimum fungsi dengan turunan} \\[8pt]
                 dE(\beta)=2(-X^Ty+X^TX\beta) & \cdots(9) \qquad \text{fungsi $E(\beta)$ diturunkan terhadap $\beta$} \\[8pt]
                 0 = 2(-X^Ty+X^TX\beta) & \cdots(10) \qquad \text{samakan dengan 0, sesuai definisi pers.(8)} \\[8pt]
                 X^TX\beta=X^Ty & \cdots(11) \qquad \text{bagi dengan 2 dan suku $-X^Ty$ pindah ke ruas kiri} \\[8pt]
                 (X^TX)^{-1}(X^TX)\beta=(X^TX)^{-1}X^Ty & \cdots(12) \qquad \text{kedua ruas dikali dengan invers $X^TX$} \\[8pt]
                 I\beta=(X^TX)^{-1}X^Ty & \cdots(13) \qquad \text{hasil kali invers menjadi matriks identitas} \\
                 \end{array}
                 """)
        st.latex(r"\beta=(X^TX)^{-1}X^Ty")

with tab2:
    st.markdown("Pada regresi linear sederhana 1 variabel, model yang digunakan adalah:")

    st.latex(r'y_i=\beta_0+\beta_1x_i+e_i')

    st.markdown("Bentuk tersebut dapat dituliskan ke dalam bentuk matriks sebagai:")
    st.latex(r'Y_{(n \times 1)} = X_{(n \times 2)} \beta_{(2 \times 1)} + e_{(n \times 1)}')
    st.latex(r'''
        \begin{bmatrix}
        y_1\\
        y_2\\
        \vdots\\
        y_n
        \end{bmatrix}
        =
        \begin{bmatrix}
        1 & x_1\\
        1 & x_2\\
        \vdots & \vdots\\
        1 & x_n
        \end{bmatrix}
        \begin{bmatrix}
        \beta_0\\
        \beta_1
        \end{bmatrix}
        +
        \begin{bmatrix}
        e_1\\
        e_2\\
        \vdots\\
        e_n
        \end{bmatrix}
        ''')

    st.markdown("Menurut metode Least Square, solusi parameter adalah sebagai berikut:")
    st.latex(r'\beta=(X^TX)^{-1}X^TY')

    st.divider()
    st.markdown(r"**Langkah pertama** adalah menghitung matriks $X^T X$")
    st.latex(r'''
        X^TX=
        \begin{bmatrix}
        1 & 1 & \cdots & 1\\
        x_1 & x_2 & \cdots & x_n
        \end{bmatrix}
        \begin{bmatrix}
        1 & x_1\\
        1 & x_2\\
        \vdots & \vdots\\
        1 & x_n
        \end{bmatrix}
        =
        \begin{bmatrix}
        n & \sum X\\
        \sum X & \sum X^2
        \end{bmatrix}
        ''')
    
    st.markdown(r"Karena $X^T X$ berukuran 2×2, inversnya dapat dihitung menggunakan rumus invers matriks:")
    with st.expander("penjelasan"):
        st.latex(r'''
            jika \qquad A =         
            \begin{bmatrix}
            a & b\\
            c & d
            \end{bmatrix}
            \qquad \qquad       
            maka \quad A^{-1} = 
            \frac{1}{ad - bc}
            \begin{bmatrix}
            d & -b\\
            -c & a
            \end{bmatrix}
            ''')
        
    st.latex(r'''
        (X^TX)^{-1}
        =
        \frac{1}{n\sum X^2-(\sum X)^2}
        \begin{bmatrix}
        \sum X^2 & -\sum X\\
        -\sum X & n
        \end{bmatrix}
        ''')

    st.divider()
    st.markdown(r"**Langkah kedua** adalah menghitung matriks $X^T Y$")
    st.latex(r'''
        X^TY=
        \begin{bmatrix}
        1 & 1 & \cdots & 1\\
        x_1 & x_2 & \cdots & x_n
        \end{bmatrix}
        \begin{bmatrix}
        y_1\\
        y_2\\
        \vdots\\
        y_n
        \end{bmatrix}
        =
        \begin{bmatrix}
        \sum Y\\
        \sum XY
        \end{bmatrix}
        ''')

    st.divider()
    st.markdown(r"**Langkah terakhir** Substitusi ke dalam persamaan $\beta=(X^TX)^{-1}X^TY$")
    st.latex(r'''
        \begin{bmatrix}
        \hat{\beta}_0\\
        \hat{\beta}_1
        \end{bmatrix}
        =
        \frac{1}{n\sum X^2-(\sum X)^2}
        \begin{bmatrix}
        \sum X^2 & -\sum X\\
        -\sum X & n
        \end{bmatrix}
        \begin{bmatrix}
        \sum Y\\
        \sum XY
        \end{bmatrix}
        ''')
    st.markdown("Lakukan Perkalian matriks dan skalar")
    st.latex(r'''
        \begin{bmatrix}
        \hat{\beta}_0\\
        \hat{\beta}_1
        \end{bmatrix}
        =
        \frac{1}{n\sum X^2-(\sum X)^2}
        \begin{bmatrix}
        \sum X^2\sum Y-\sum X\sum XY\\
        n\sum XY-\sum X\sum Y
        \end{bmatrix}
        ''')
    
    st.latex(r'''
        \begin{bmatrix}
        \hat{\beta}_0\\[8pt]
        \hat{\beta}_1
        \end{bmatrix}
        =
        \begin{bmatrix}
        \frac{\sum X^2\sum Y-\sum X\sum XY}{n\sum X^2-(\sum X)^2}\\[8pt]
        \frac{n\sum XY-\sum X\sum Y}{n\sum X^2-(\sum X)^2}
        \end{bmatrix}
        ''')

    st.markdown("Dengan demikian maka diperoleh:")
    col1, col2, = st.columns(2, gap='xxsmall')
    col1.latex(r'''
        \hat{\beta}_0
        =
        \frac{\sum X^2\sum Y-\sum X\sum XY}
        {n\sum X^2-(\sum X)^2}
        ''')

    col2.latex(r'''
        \hat{\beta}_1
        =
        \frac{n\sum XY-\sum X\sum Y}
        {n\sum X^2-(\sum X)^2}
        ''')

with tab3:
    format_sitasi = st.selectbox(
        "Pilih Format Sitasi",
        ["APA", "IEEE", "MLA"]
    )

    if format_sitasi == "APA":
        st.markdown("""
    ### Referensi (APA 7th)
    1. Roustaei, N. (2024). Application and interpretation of linear-regression analysis. 
                    *Medical Hypothesis, Discovery & Innovation in Ophthalmology*, 13(3), 151–159.

    2. Penn State University. (n.d.). *A Matrix Formulation of the Multiple Regression Model*.
                    https://online.stat.psu.edu/stat462/node/132/

    3. Jeong, C., & Byon, E. (2024). 
                    Calibration of building energy computer models via bias-corrected iteratively reweighted least squares method. 
                    *Applied Energy*, 360, 122753.

    4. Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021).
                    *Introduction to Linear Regression Analysis* (6th ed.).
                    Wiley.

    5. Weisberg, S. (2014).
                    *Applied Linear Regression* (4th ed.).
                    Wiley.
    """)

    elif format_sitasi == "IEEE":
        st.markdown("""
    ### Referensi (IEEE)
    [1] N. Roustaei, “Application and interpretation of linear-regression analysis,” 
    *Medical Hypothesis, Discovery & Innovation in Ophthalmology*, 
    vol. 13, no. 3, pp. 151–159, Oct. 2024
    
    [2] Penn State University,
    “A Matrix Formulation of the Multiple Regression Model.”
    [Online]. Available:
    https://online.stat.psu.edu/stat462/node/132/
                    
    [3] C. Jeong and E. Byon, 
    “Calibration of building energy computer models via bias-corrected iteratively reweighted least squares method,” 
    *Applied Energy*, vol. 360, Art. no. 122753, 2024
                    
    [4] D. C. Montgomery, E. A. Peck, and G. G. Vining,
    *Introduction to Linear Regression Analysis*, 6th ed.
    Hoboken, NJ, USA: Wiley, 2021.

    [5] S. Weisberg,
    *Applied Linear Regression*, 4th ed.
    Hoboken, NJ, USA: Wiley, 2014.
    """)

    elif format_sitasi == "MLA":
        st.markdown("""
    ### Referensi (MLA 9th)

    1. Roustaei, Narges. “Application and Interpretation of Linear-Regression Analysis.” 
                    *Medical Hypothesis, Discovery & Innovation in Ophthalmology*, 
                    vol. 13, no. 3, 2024, pp. 151–159.
    2. Penn State University.
                    "A Matrix Formulation of the Multiple Regression Model."
                    STAT 462,
                    https://online.stat.psu.edu/stat462/node/132/.

    3. Jeong, Chanyoung, and Eunshin Byon. 
                    “Calibration of Building Energy Computer Models via Bias-Corrected Iteratively Reweighted Least Squares Method.” 
                    *Applied Energy*, vol. 360, 2024, article 122753. Elsevier
                    
    4. Montgomery, Douglas C., Elizabeth A. Peck, and G. Geoffrey Vining.
                    *Introduction to Linear Regression Analysis*.
                    6th ed., Wiley, 2021.

    5. Weisberg, Sanford.
                    *Applied Linear Regression*.
                    4th ed., Wiley, 2014.
    """)

with tab4:
    colSetting, colVisual = st.columns([0.25,0.75], gap='xsmall', border=True)

    with colSetting:
        st.header("Dataset")
        sumber_data = st.radio(" ",["Data Sampel", "Data Custom"], horizontal=True, label_visibility="collapsed")

        if sumber_data == "Data Custom":
            with st.form("generate_form", border=False):
                col1, col2, col3 = st.columns(3)
                n = col1.number_input("Sampel", min_value=10)
                p = col2.number_input("Variabel", min_value=1, max_value=10, value=3)
                outlier = col3.number_input("Outlier", min_value=0, max_value=10)
                noise = st.slider("Noise (%)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

                tabX, tabY = st.tabs(['Generate X', 'Generate Y'])
                with tabX:
                    tab_x = pd.DataFrame({
                        " ": [f"x{i+1}" for i in range(p)],
                        "Distribusi": ["Linear Space"] * p,
                        "a": [-5] * p,
                        "b": [5] * p
                    })
                    st.caption("a dan b adalah parameter untuk distribusi,   a sebagai min/mean(μ) dan b sebagai max/standar deviasi(σ)")
                    tab_x = st.data_editor(tab_x, hide_index=True, key="tab_x",
                                column_config={
                                    "Distribusi": st.column_config.SelectboxColumn(
                                        "Distribusi", options=["Linear Space", "Uniform", "Normal", "Integer"]
                                    )
                                }
                            )

                with tabY:
                    equation = pd.DataFrame({
                        "Aktif": [True] * p,
                        "Koefisien": [1.0] * p,
                        "Pangkat": [1] * p,
                        "Fungsi": ["None"] * p,
                    }, index=[f"x{i+1}" for i in range(p)])

                    equation = st.data_editor(equation, hide_index=False, key='equation',
                                column_config={
                                    "Fungsi": st.column_config.SelectboxColumn(
                                        options=["None", "sin", "cos", "tan", "exp", "log", "sqrt"]
                                    )
                                }
                            )

                    fungsi = {
                        "None": "{}",
                        "sin": r"\sin({})",
                        "cos": r"\cos({})",
                        "tan": r"\tan({})",
                        "exp": r"e^{{{}}}",
                        "log": r"\log({})",
                        "sqrt": r"\sqrt{{{}}}",
                    }

                    if not equation["Aktif"].any():
                        st.warning("Minimal satu variabel harus aktif.")
                        st.stop()

                    mask = equation["Aktif"]
                    latex_list = []

                    for var_name, r in equation.loc[mask].iterrows():
                        var = f"x_{{{var_name[1:]}}}"
                        if r["Pangkat"] > 1:
                            var += f"^{{{int(r['Pangkat'])}}}"

                        term = fungsi[r["Fungsi"]].format(var)

                        if r["Koefisien"] != 1:
                            term = f"{f_num(r['Koefisien'])}{term}"

                        latex_list.append(f"{term} + ")

                    latex = " ".join(latex_list)
                    const = st.number_input("kostanta", value=1, help="jika tidak ingin menggunakan konstanta, maka ubah menjadi 0")

                    st.latex(rf"y={latex} {const:g} + \epsilon")

                generate = st.form_submit_button("Generate Data", type="primary", width='stretch')

            if generate:
                st.warning("Ini masih tahap percobaan")
            
            st.stop()
        else:
            data = st.selectbox("Pilih Data", Dataset.keys())
            st.session_state.df = Dataset[data]['data']
            st.latex(Dataset[data]["latex"][0], help="Data tersebut dibentuk berdasarkan persamaan ini")
            st.latex(Dataset[data]["latex"][1], help="Sejatinya 'RMSE' adalah standar deviasi (σ) dari data asli, karena variansi error (σ²) adalah √MSE atau RMSE, perhatikan dan bandingkan angkanya (R² dan MSE tidak berlaku untuk data circle)")

            st.session_state.df = st.data_editor(
                st.session_state.df,
                height=422
            )

    with colVisual:
        @st.fragment
        def least_square_simulation():
            st.title("Least Square Simulation")

            df_source = st.session_state.get("df")
            if df_source is None or df_source.empty:
                return

            df = df_source.copy()

            obj_cols = df.select_dtypes(include="object").columns
            for col in obj_cols:
                df[col] = df[col].astype("category").cat.codes

            poly_degree = {
                "Polinomial (kuadrat)": 2,
                "Polinomial (kubik)": 3,
            }
            is_circle = data == "Circle"
            is_poly = data in poly_degree
            is_simple = data in {"Linear", "Circle"} or is_poly

            colA, colB = st.columns([0.7, 0.3])

            with colA:
                if is_simple:
                    x = "x" if "x" in df.columns else df.columns[0]
                    y = "y" if "y" in df.columns else [c for c in df.columns if c != x][0]
                    st.caption(f"Sumbu yang dipakai: **{x}** sebagai X dan **{y}** sebagai Y.")
                else:
                    col1, col2 = st.columns(2)

                    x = col1.selectbox("Variabel X", df.columns)
                    opsi_y = [c for c in df.columns if c != x]
                    if not opsi_y:
                        return

                    default_y = "y" if "y" in opsi_y else opsi_y[0]
                    y = col2.selectbox("Variabel Y", opsi_y, index=opsi_y.index(default_y))

            x_arr = df[x].to_numpy(dtype=float, copy=False)
            y_arr = df[y].to_numpy(dtype=float, copy=False)
            n = len(df)

            if is_circle:
                A = np.c_[x_arr, y_arr, np.ones(n)]
                b = -(x_arr**2 + y_arr**2)

            elif data == "Linear":
                A = np.c_[np.ones(n), x_arr]
                b = y_arr

            elif is_poly:
                degree = poly_degree[data]
                A = np.column_stack([np.ones(n)] + [x_arr**p for p in range(1, degree + 1)])
                b = y_arr

            else:
                X = df.drop(columns=[y])
                A = np.c_[np.ones(n), X.to_numpy(dtype=float, copy=False)]
                b = y_arr

            hasil = least_square(A, b)
            beta = np.asarray(hasil["beta"], dtype=float)

            with colB:
                st.space("medium")
                st.header("Metric", text_alignment="center")

                colC, colD = st.columns(2)
                colC.metric("SSE", f"{hasil['SSE']:.2f}", border=True)
                colC.metric("MSE", f"{hasil['MSE']:.2f}", border=True)
                colD.metric("RMSE", f"{hasil['RMSE']:.2f}", border=True)
                colD.metric("R²", f"{hasil['R2']:.2f}", border=True)

                df_beta = pd.DataFrame(
                    {"Parameter": [f"β{i}" for i in range(len(beta))],
                     "Estimasi": beta,
                     "Asli": Dataset[data]["beta"]}
                )

                st.dataframe(df_beta, hide_index=True, width="stretch")
                if data == "Circle":
                    st.button("INFO", on_click=penjelas_circle)
                                

            latex = _build_latex(beta, data)

            df_plot = df.copy()
            df_plot["Residual"] = np.asarray(hasil["residual"], dtype=float)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_plot[x],
                    y=df_plot[y],
                    mode="markers",
                    marker=dict(size=10, color="#2563eb", line=dict(color="white", width=1)),
                    showlegend=False,
                )
            )

            if is_circle:
                D, E, F = beta
                xc, yc = -D / 2, -E / 2
                radicand = xc**2 + yc**2 - F
                r = np.sqrt(max(radicand, 0.0))

                t_curve = np.linspace(0, 2 * np.pi, 360)
                x_fit = xc + r * np.cos(t_curve)
                y_fit = yc + r * np.sin(t_curve)

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name="Fitting Circle",
                        line=dict(color="red", width=3),
                        showlegend=False,
                    )
                )
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

            else:
                x_curve = np.linspace(df[x].min(), df[x].max(), 500)

                if data == "Linear":
                    y_curve = beta[0] + beta[1] * x_curve

                elif is_poly:
                    degree = poly_degree[data]
                    y_curve = np.full_like(x_curve, beta[0], dtype=float)
                    for p in range(1, degree + 1):
                        y_curve += beta[p] * x_curve**p

                else:
                    X = df.drop(columns=[y])
                    curve_df = pd.DataFrame(
                        {
                            col: np.full_like(x_curve, float(X[col].mean()), dtype=float)
                            for col in X.columns
                        }
                    )
                    curve_df[x] = x_curve

                    y_curve = np.full_like(x_curve, beta[0], dtype=float)
                    for i, col in enumerate(X.columns, start=1):
                        y_curve += beta[i] * curve_df[col].to_numpy(dtype=float, copy=False)

                fig.add_trace(
                    go.Scatter(
                        x=x_curve,
                        y=y_curve,
                        mode="lines",
                        name="Fitting",
                        line=dict(color="red", width=3),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                height=500,
                template="simple_white",
                xaxis=dict(title=x, showgrid=True, constrain="domain", ticks="outside"),
                yaxis=dict(title=y, showgrid=True, ticks="outside"),
            )

            x_res = df_plot[x].to_numpy(dtype=float, copy=False)
            r_res = df_plot["Residual"].to_numpy(dtype=float, copy=False)

            fig_res = go.Figure()
            fig_res.add_trace(
                go.Scatter(
                    x=x_res,
                    y=r_res,
                    mode="markers",
                    marker=dict(size=10, color="#2563eb", line=dict(color="white", width=1)),
                    showlegend=False,
                )
            )

            fig_res.add_trace(
                go.Scatter(
                    x=np.repeat(x_res, 3),
                    y=np.column_stack(
                        [
                            np.zeros_like(r_res),
                            r_res,
                            np.full_like(r_res, np.nan, dtype=float),
                        ]
                    ).ravel(),
                    mode="lines",
                    line=dict(color="gray", dash="dot", width=2),
                    showlegend=False,
                )
            )
            fig_res.add_hline(y=0, line_color="red", line_width=2)
            fig_res.update_layout(
                height=500,
                template="simple_white",
                xaxis=dict(title=x, showgrid=True, ticks="outside"),
                yaxis=dict(title="Residual", showgrid=True, ticks="outside", rangemode="tozero"),
            )

            fig_hist = px.histogram(
                df_plot,
                x="Residual",
                nbins=15,
                opacity=0.8,
                template="simple_white",
            )
            fig_hist.update_layout(height=500, template="simple_white")

            with colA:
                st.latex(latex)
                tab1, tab2, tab3, tab4 = st.tabs(["Regresi", "Residual", "Distribusi Residual", "Visual Matrix"])

                with tab1:
                    st.plotly_chart(fig, width="stretch")

                with tab2:
                    st.plotly_chart(fig_res, width="stretch")

                with tab3:
                    st.plotly_chart(fig_hist, width="stretch")

                with tab4:
                    st.subheader("Visual Matrix")

                    steps = _matrix_visual_story(data)

                    for i, step in enumerate(steps):
                        with st.container(border=True):
                            st.markdown(f"**{step['title']}**")

                            if step["kind"] == "single":
                                st.latex(step["latex"])

                            elif step["kind"] == "triplet":
                                colA, colB, colC = st.columns([4, 2, 3], gap="xxsmall")
                                colA.latex(step["A"])
                                colB.latex(step["beta"])
                                colC.latex(step["b"])
                            
                                st.latex(step["prod"])

                        if i < len(steps) - 1:
                            st.markdown(
                                "<div style='text-align:center;font-size:26px;line-height:1;'>↓</div>",
                                unsafe_allow_html=True
                            )

        least_square_simulation()
