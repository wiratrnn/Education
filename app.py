import streamlit as st

st.set_page_config(layout='wide')

st.title("Metode Least Square")
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

tab1, tab2, tab3, tab4 = st.tabs(['Penurunan Rumus', 'Penyederhanaan Rumus', '📚 Referensi', 'Simulasi'])
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
    st.latex(r'''
        \hat{\beta}_0
        =
        \frac{\sum X^2\sum Y-\sum X\sum XY}
        {n\sum X^2-(\sum X)^2}
        ''')

    st.latex(r'''
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
    st.info("Masih dalam tahap pengembangan")