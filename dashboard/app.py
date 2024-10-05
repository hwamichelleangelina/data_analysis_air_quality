import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Dashboard Kualitas Udara Beijing", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        options=["Gambaran Set Data Kualitas Udara Stasiun Beijing", 
                                 "Statistik Analisis Kualitas Udara Stasiun", 
                                 "Korelasi Polutan dan Analisis Lokasi Stasiun", 
                                 "Distribusi Kualitas Udara di Stasiun Beijing", 
                                 "Konsentrasi Polutan dalam Sehari di Beijing"],
                        index=0)

icons = {
    "Gambaran Set Data Kualitas Udara Stasiun Beijing": "📊",
    "Statistik Analisis Kualitas Udara Stasiun": "📈",
    "Korelasi Polutan dan Analisis Lokasi Stasiun": "📍",
    "Distribusi Kualitas Udara di Stasiun Beijing": "🗺️",
    "Konsentrasi Polutan dalam Sehari di Beijing": "⏳"
}

st.title(f"{page}")

aotizhongxin_df = pd.read_csv('aotizhongxin.csv')
changping_df = pd.read_csv('changping.csv')
dingling_df = pd.read_csv('dingling.csv')
dongsi_df = pd.read_csv('dongsi.csv')
guanyuan_df = pd.read_csv('guanyuan.csv')
gucheng_df = pd.read_csv('gucheng.csv')
huairou_df = pd.read_csv('huairou.csv')
nongzhanguan_df = pd.read_csv('nongzhanguan.csv')
shunyi_df = pd.read_csv('shunyi.csv')
tiantan_df = pd.read_csv('tiantan.csv')
wanliu_df = pd.read_csv('wanliu.csv')
wanshouxigong_df = pd.read_csv('wanshouxigong.csv')

dfs = [aotizhongxin_df, changping_df, dingling_df, dongsi_df, guanyuan_df, gucheng_df,
       huairou_df, nongzhanguan_df, shunyi_df, tiantan_df, wanliu_df, wanshouxigong_df]
all_stations_df = pd.concat(dfs)

if 'wd' in all_stations_df.columns:
    all_stations_df = all_stations_df.drop(columns=['wd'])

all_stations_df_clean = all_stations_df.dropna()
all_stations_df['year'] = all_stations_df['year'].astype(int)

station_stats = all_stations_df_clean.groupby('station').agg({
    'PM2.5': ['mean', 'median', 'min', 'max'],
    'PM10': ['mean', 'median', 'min', 'max'],
    'SO2': ['mean', 'median', 'min', 'max'],
    'NO2': ['mean', 'median', 'min', 'max'],
    'CO': ['mean', 'median', 'min', 'max'],
    'O3': ['mean', 'median', 'min', 'max']
}).reset_index()

station_stats.columns = ['station'] + [f'{pollutant}_{stat}' for pollutant, stat in station_stats.columns[1:]]

def find_max_pollutant_per_station(df, pollutant):
    max_pollutant = df.loc[df.groupby('station')[pollutant].idxmax()]
    return max_pollutant[['station', 'year', 'month', 'day', 'hour', pollutant]]

max_pm25 = find_max_pollutant_per_station(all_stations_df, 'PM2.5')
max_pm10 = find_max_pollutant_per_station(all_stations_df, 'PM10')
max_so2 = find_max_pollutant_per_station(all_stations_df, 'SO2')
max_no2 = find_max_pollutant_per_station(all_stations_df, 'NO2')
max_co = find_max_pollutant_per_station(all_stations_df, 'CO')
max_o3 = find_max_pollutant_per_station(all_stations_df, 'O3')

def format_max_pollutant(pollutant_data, column):
    station = pollutant_data['station']
    date_str = f"{int(pollutant_data['day'])}-{int(pollutant_data['month'])}-{int(pollutant_data['year'])} {int(pollutant_data['hour']):02d}:00"
    value = pollutant_data[column]
    return f"{station} pada tanggal {date_str} dengan konsentrasi {value:.2f}"

all_stations_df['year'] = all_stations_df['year'].astype(str)
mean_polutan_per_station_yearly = all_stations_df.groupby(['station', 'year']).mean().reset_index()

if page == "Gambaran Set Data Kualitas Udara Stasiun Beijing":
    st.header("Overview Dataset")
    st.write("Dataset ini terdiri dari berbagai data kualitas udara dari beberapa stasiun di Beijing. Di bawah ini adalah beberapa informasi tentang dataset tersebut:")
    
    st.subheader("Data Gabungan Semua Stasiun:")
    formatted_data = []
    for _, row in all_stations_df_clean.iterrows():
        formatted_data.append({
            "station": row['station'],
            "year": f"{row['year']}",
            "month": row['month'],
            "day": row['day'],
            "hour": row['hour'],
            "PM2.5": row['PM2.5'],
            "PM10": row['PM10'],
            "SO2": row['SO2'],
            "NO2": row['NO2'],
            "CO": row['CO'],
            "O3": row['O3']
        })

    st.write(pd.DataFrame(formatted_data))
    
    st.subheader("Polusi Tertinggi:")
    st.write("Berikut adalah data polusi tertinggi untuk setiap polutan di stasiun Beijing:")

    st.write("###### Polusi Tertinggi PM2.5")
    st.write(format_max_pollutant(max_pm25.iloc[0], 'PM2.5'))

    st.write("###### Polusi Tertinggi PM10")
    st.write(format_max_pollutant(max_pm10.iloc[0], 'PM10'))

    st.write("###### Polusi Tertinggi SO2")
    st.write(format_max_pollutant(max_so2.iloc[0], 'SO2'))

    st.write("###### Polusi Tertinggi NO2")
    st.write(format_max_pollutant(max_no2.iloc[0], 'NO2'))

    st.write("###### Polusi Tertinggi CO")
    st.write(format_max_pollutant(max_co.iloc[0], 'CO'))

    st.write("###### Polusi Tertinggi O3")
    st.write(format_max_pollutant(max_o3.iloc[0], 'O3'))

    st.write("Terlihat nilai-nilai polutan tertinggi di berbagai stasiun terjadi di hari-hari yang sama. Karena seluruh stasiun berada di Beijing, polutan akan cenderung sama kualitasnya. Oleh karena itu, pada saat nilai-nilai polutan tertinggi terjadi, seluruh stasiun dalam Beijing sedang memiliki kualitas udara yang buruk.")

if page == "Statistik Analisis Kualitas Udara Stasiun":
    st.header("Statistik Analisis Kualitas Udara Stasiun")
    st.write("Tampilkan statistik tambahan atau visualisasi untuk analisis lebih mendalam.")

    station_stats.columns = ['station', 'PM2.5_mean', 'PM2.5_median', 'PM2.5_min', 'PM2.5_max',
                             'PM10_mean', 'PM10_median', 'PM10_min', 'PM10_max',
                             'SO2_mean', 'SO2_median', 'SO2_min', 'SO2_max',
                             'NO2_mean', 'NO2_median', 'NO2_min', 'NO2_max',
                             'CO_mean', 'CO_median', 'CO_min', 'CO_max',
                             'O3_mean', 'O3_median', 'O3_min', 'O3_max']

    st.write(station_stats)

    station_avg = all_stations_df_clean.groupby('station').mean()

    plt.figure(figsize=(14, 4))
    station_avg[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].plot(kind='bar', figsize=(8, 4))
    plt.title('Rata-rata Polusi Udara per Stasiun')
    plt.ylabel('Konsentrasi Polutan')
    plt.grid(True)

    st.pyplot(plt)

    st.write("Dalam grafik di atas, kita dapat melihat rata-rata konsentrasi polutan yang diukur di berbagai stasiun pemantauan udara di Beijing. Meskipun terdapat variasi di antara nilai-nilai spesifik untuk setiap polutan, terlihat bahwa rata-rata nilai polutan di setiap stasiun memiliki tingkatan yang relatif serupa.")
    st.write("Setiap stasiun menunjukkan pola yang konsisten dalam hal konsentrasi polutan seperti PM2.5, PM10, SO2, NO2, CO, dan O3. Dengan kata lain, meskipun ada perbedaan lokal yang mungkin disebabkan oleh faktor-faktor seperti kepadatan lalu lintas, aktivitas industri, dan kondisi cuaca, data menunjukkan bahwa polusi udara di Beijing tidak terkonsentrasi hanya di satu area tertentu.")
    st.write("Seluruh grafik batang yang ditampilkan juga menunjukkan bentuk yang mirip, yang menunjukkan bahwa tingkat polusi cenderung merata di seluruh stasiun. Ini bisa diartikan bahwa warga di berbagai bagian kota berpotensi terpapar pada tingkat polusi udara yang serupa, dan masalah polusi udara ini tidak hanya terbatas pada area tertentu saja.")
    st.write("Hal ini menunjukkan bahwa langkah-langkah untuk mengatasi polusi udara perlu diterapkan secara menyeluruh di seluruh kota, bukan hanya fokus pada area dengan tingkat polusi tertinggi. Memahami pola persebaran ini sangat penting untuk perencanaan kebijakan dan tindakan yang efektif guna mengurangi polusi udara dan melindungi kesehatan masyarakat.")

elif page == "Korelasi Polutan dan Analisis Lokasi Stasiun":
    st.write("Halaman ini akan menganalisis korelasi antara polutan dan lokasi stasiun di Beijing.")

    plt.figure(figsize=(5, 2))
    sns.heatmap(all_stations_df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].corr(), annot=True, cmap='coolwarm')
    plt.title('Korelasi antara Polutan di Semua Stasiun')
    st.pyplot(plt)
    plt.clf()

    st.write("Terlihat potensi korelasi antar polutan dari gambar di atas.")

    st.write("**Korelasi Positif:**")
    st.write("- PM2.5 dan PM10 memiliki korelasi yang sangat kuat; kedua jenis partikel ini sering muncul dengan konsentrasi mirip.")
    st.write("- PM2.5 dan CO serta PM10 dan CO juga memiliki korelasi yang tinggi; ketika konsentrasi partikel-partikel ini tinggi, CO juga cenderung meningkat.")
    st.write("- NO2 memiliki korelasi sedang dengan PM2.5, PM10, dan CO; NO2 kadang-kadang sejalan dengan kenaikan PM dan CO, tetapi tidak selalu.")
    st.write("- SO2 memiliki korelasi yang sedang dengan polutan lainnya, kecuali O3.")

    st.write("**Korelasi Negatif:**")
    st.write("- O3 menunjukkan korelasi negatif dengan sebagian besar polutan lainnya; ketika konsentrasi O3 meningkat, konsentrasi polutan lain cenderung menurun.")
    st.write("- O3 ada korelasi sedang dengan NO2, tetapi sisanya berkorelasi negatif yang cukup lemah.")

    st.write("Di bawah ini adalah ambang batas yang digunakan untuk mengkategorikan polutan udara di China:")
    threshold = {'PM2.5': 75, 'PM10': 150, 'SO2': 150, 'NO2': 200, 'CO': 800, 'O3': 160}
    st.write(threshold)

    for polutan in threshold:
        all_stations_df[f'{polutan}_high'] = all_stations_df[polutan] > threshold[polutan]

    rfm = all_stations_df.groupby('station').agg({
        'day': 'max',
        'PM2.5_high': 'sum',
        'PM2.5': 'max'
    }).rename(columns={'day': 'Recency', 'PM2.5_high': 'Frequency', 'PM2.5': 'Monetary'}).reset_index()

    st.write("Data RFM (Recency, Frequency, Monetary):")
    st.write(rfm)

    st.write("- Hasil recency menunjukkan semua stasiun memiliki nilai recency yang sama, yaitu 31. Ini menunjukkan bahwa data kualitas udara di semua stasiun diambil pada periode yang sama, sehingga analisis perbandingan antar stasiun akan konsisten.")
    st.write("- Stasiun Dongsi memiliki frekuensi tertinggi (14965), diikuti oleh stasiun Guanyuan (14461) dan Gucheng (14555). Hal ini menunjukkan bahwa stasiun-stasiun ini memiliki tingkat pemantauan yang tinggi dan harus diperhatikan.")
    st.write("- Stasiun Dingling memiliki frekuensi terendah (11120), menunjukkan bahwa data dari stasiun ini mungkin kurang dapat diandalkan untuk analisis lebih lanjut.")
    st.write("- Dari sisi monetary, stasiun Dongsi juga menunjukkan nilai tertinggi (265), diikuti oleh Guanyuan (253) dan Gucheng (254). Konsentrasi polutan di Dongsi lebih tinggi sehingga indikasi kualitas udara di stasiun-stasiun ini cenderung lebih buruk.")
    st.write("- Stasiun Huairou mencatat nilai monetary terendah (222). Hal ini menunjukkan bahwa kualitas udara di daerah ini mungkin lebih baik dibandingkan dengan stasiun-stasiun lainnya.")

    average_pollutants = all_stations_df.groupby('station')[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()

    boundaries = average_pollutants.quantile([0.33, 0.66])

    def categorize(value, low, high):
        if value < low:
            return 'Rendah'
        elif value < high:
            return 'Sedang'
        else:
            return 'Tinggi'

    categorized_pollutants = average_pollutants.apply(
        lambda x: pd.Series(
            [categorize(val, boundaries.loc[0.33, col], boundaries.loc[0.66, col]) for val, col in zip(x, average_pollutants.columns)],
            index=average_pollutants.columns
        ),
        axis=1
    )

    def group_station(row):
        high_pollutants = row.value_counts().get('Tinggi', 0)
        if high_pollutants >= 4:
            return 'Sangat Tercemar'
        elif high_pollutants >= 2:
            return 'Tercemar'
        else:
            return 'Sedang'

    categorized_pollutants['Grup'] = categorized_pollutants.apply(group_station, axis=1)

    group_counts = categorized_pollutants['Grup'].value_counts()
    plt.figure(figsize=(8, 5))
    group_counts.plot(kind='bar', color=['yellow', 'orange', 'red'])
    plt.title('Distribusi Stasiun Berdasarkan Kategori Kualitas Udara')
    plt.xlabel('Grup Stasiun')
    plt.ylabel('Jumlah Stasiun')
    st.pyplot(plt)
    plt.clf()

    st.write("Grafik di atas menunjukkan distribusi jumlah stasiun berdasarkan kategori kualitas udara yang ditentukan. "
            "Stasiun-stasiun ini dikelompokkan menjadi tiga kategori: 'Rendah', 'Sedang', dan 'Tercemar', "
            "dengan 'Sangat Tercemar' menandakan bahwa lebih dari empat polutan diukur berada di atas ambang batas. "
            "Dengan analisis ini, kita dapat lebih memahami sebaran kualitas udara di Beijing dan mengidentifikasi "
            "area yang membutuhkan perhatian lebih lanjut.")

elif page == "Distribusi Kualitas Udara di Stasiun Beijing":
    st.write("Halaman ini akan menampilkan distribusi kualitas udara dari beberapa stasiun di Beijing.")

    mean_polutan_per_station_yearly_2017 = mean_polutan_per_station_yearly[mean_polutan_per_station_yearly['year'] == '2017']
    average_pollutants = mean_polutan_per_station_yearly_2017[['station', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].copy()
    average_pollutants_melted = average_pollutants.melt(id_vars='station', var_name='pollutant', value_name='concentration')
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    fig, axes = plt.subplots(1, 6, figsize=(14, 35))
    axes = axes.flatten()

    for i, pollutant in enumerate(pollutants):
        pollutant_data = average_pollutants_melted[average_pollutants_melted['pollutant'] == pollutant]

        sns.scatterplot(
            data=pollutant_data,
            x='pollutant',
            y='concentration',
            size='concentration',
            sizes=(150, 500),
            hue='concentration',
            palette='coolwarm',
            alpha=0.6,
            legend=False,
            ax=axes[i]
        )

        for index, row in pollutant_data.iterrows():
            axes[i].annotate(
                row['station'],
                (row['pollutant'], row['concentration']),
                textcoords="offset points",
                xytext=(-10, 0),
                ha='center',
                fontsize=8
            )

        axes[i].set_title(f'{pollutant} (Tahun 2017)', fontsize=10)
        axes[i].set_ylabel('Konsentrasi', fontsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    st.pyplot(fig)

    st.write("""
    Grafik menunjukkan distribusi rata-rata konsentrasi polutan utama (PM2.5, PM10, SO2, NO2, CO, dan O3) di berbagai stasiun di Beijing pada tahun 2017 (tahun terakhir di dataset yang disediakan). Ukuran dan intensitas warna lingkaran mencerminkan tingkat konsentrasi polusi, dengan bubble yang lebih besar dan berwarna merah menunjukkan tingkat polusi yang lebih tinggi.

    - Stasiun Dongsi, Wanshouxigong, dan Guanyuan menunjukkan konsentrasi yang lebih tinggi baik untuk PM2.5 maupun PM10, ditandai dengan bubble yang lebih besar dan berwarna merah tua. Polutan-partikulat ini dikenal berbahaya bagi kesehatan, khususnya pernapasan. Hal ini menunjukkan bahwa inti kota mengalami tingkat polusi partikulat 2.5 dan 10 yang lebih tinggi, kemungkinan disebabkan oleh lalu lintas, konstruksi, dan aktivitas industri di sekitar stasiun.
    - Stasiun Shunyi memiliki konsentrasi SO2 tertinggi, diikuti oleh stasiun seperti Aotizhongxin dan Guanyuan yang juga menunjukkan peningkatan level SO2. Shunyi menjadi penyumbang SO2 tertinggi di kota Beijing.
    - Stasiun Wanliu dan Guanyuan memiliki konsentrasi NO2 tertinggi, yang umumnya terkait dengan emisi kendaraan dan aktivitas industri.
    - Stasiun Shunyi dan Tiantan menunjukkan konsentrasi CO yang lebih tinggi, polutan ini terutama dihasilkan oleh kendaraan dan pembakaran bahan bakar fosil. Wilayah-wilayah ini menunjukkan adanya kemungkinan aktivitas lalu lintas yang terlalu padat.
    - Dingling dan Dongsi menunjukkan konsentrasi ozon tertinggi.

    Oleh karena itu,
    - Stasiun Dongsi, Wanshouxigong, Guanyuan cenderung memiliki konsentrasi yang lebih tinggi untuk PM2.5, PM10, dan NO2, kemungkinan disebabkan oleh lalu lintas yang padat, populasi tinggi, serta aktivitas industri. Area ini harus menjadi prioritas utama untuk pengendalian polusi, khususnya terkait emisi kendaraan dan pabrik-pabrik industri. Hal ini kemungkinan terjadi karena stasiun berada di pusat kota Beijing.
    - Wilayah stasiun Shunyi dan Dingling menunjukkan level SO2, CO, dan O3 yang lebih tinggi, mengindikasikan bahwa sumber polusi di daerah ini (seperti aktivitas industri dan transportasi) juga memerlukan pengawasan ketat dan regulasi untuk mengurangi emisi berbahaya.

    Solusi untuk masalah ini dapat diimplementasikan:
    - Fokus pada stasiun-stasiun dengan level PM2.5, PM10, dan NO2 yang tinggi memerlukan kontrol emisi kendaraan yang lebih ketat, promosi transportasi umum, serta pengurangan emisi industri.
    - Pengendalian di kawasan industri memerlukan pengawasan ketat terhadap polusi industri serta penggunaan energi untuk mengurangi level SO2 dan CO.
    - Kebijakan yang berfokus pada pengurangan emisi harus diterapkan, terutama selama bulan-bulan musim panas ketika pembentukan ozon tinggi (efek rumah kaca).
    """)

elif page == "Konsentrasi Polutan dalam Sehari di Beijing":
    st.write("Halaman ini akan menampilkan perubahan konsentrasi polutan sepanjang hari di Beijing.")

    all_stations_df_clean_all_time = all_stations_df.drop(columns=['station'])
    all_stations_df_clean_all_time['year'] = pd.to_numeric(all_stations_df_clean_all_time['year'], errors='coerce')

    hourly_avg = all_stations_df_clean_all_time.groupby('hour').mean().reset_index()

    plt.figure(figsize=(14, 5))
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']
    colors = ['b', 'g', 'r', 'c', 'y']

    lines = []
    labels = []

    for i, pollutant in enumerate(pollutants):
        line, = plt.plot(hourly_avg['hour'], hourly_avg[pollutant], label=pollutant, color=colors[i])
        lines.append(line)
        labels.append(pollutant)

    plt.xlabel('Jam (Hour)', fontsize=12)
    plt.ylabel('Konsentrasi Polutan (non-CO)', fontsize=12)

    ax2 = plt.gca().twinx()
    line_co, = ax2.plot(hourly_avg['hour'], hourly_avg['CO'], label='CO', color='m')
    ax2.set_ylabel('Konsentrasi CO', color='m')

    lines.append(line_co)
    labels.append('CO')

    plt.legend(lines, labels, loc='upper left', title='Polutan')

    plt.axvspan(9, 17, color='blue', alpha=0.2, label='Jam Kerja')
    plt.axvspan(6, 10, color='red', alpha=0.2, label='Peralihan')
    plt.axvspan(16, 20, color='red', alpha=0.2, label='Peralihan')
    plt.axvspan(0, 9, color='gray', alpha=0.2, label='Jam Non-kerja Pagi')
    plt.axvspan(17, 23, color='gray', alpha=0.2, label='Jam Non-kerja Sore')

    plt.title('Polutan Berdasarkan Waktu Sehari di Seluruh Stasiun', fontsize=16)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    st.pyplot(plt)

    st.write("""
    Visualisasi grafik garis di atas menunjukkan konsentrasi rata-rata polutan (PM2.5, PM10, SO2, NO2, O3, dan CO) per jam di seluruh stasiun pemantauan. 
    - Garis berwarna menunjukkan konsentrasi masing-masing polutan sepanjang hari, dengan CO ditampilkan pada sumbu y kedua.
    - Area yang ditandai menunjukkan jam kerja, peralihan, dan jam non-kerja, memberikan konteks tentang fluktuasi polusi sepanjang hari.

    Grafik di atas menampilkan konsentrasi berbagai polutan udara di stasiun-stasiun di Beijing pada berbagai jam dalam satu hari.

    **Jam Kerja (09:00 - 17:00):**
    - Pada jam kerja, konsentrasi polutan seperti PM10, PM2.5, NO2, dan SO2 cenderung lebih tinggi. Ini kemungkinan besar disebabkan oleh peningkatan aktivitas kendaraan bermotor sekitar stasiun dan hasil dari aktivitas industri yang mulai berjalan.
    - Konsentrasi CO menunjukkan tren penurunan selama jam kerja, yang mungkin mengindikasikan bahwa sumber emisi CO lebih aktif pada pagi hari sebelum jam kerja atau pada malam hari setelah jam kerja.
    - O3 cenderung mengalami penurunan selama jam kerja. Hal ini mungkin disebabkan oleh reaksi kimia atmosfer yang kompleks yang dipengaruhi oleh sinar matahari, suhu, dan keberadaan polutan lainnya seperti NO2 pada saat kota sibuk beraktivitas.

    **Jam Non-kerja (18:00 - 07:00):**
    - Pada periode istirahat, terdapat kenaikan tajam konsentrasi CO yang dimulai sekitar pukul 18:00, mencapai puncaknya sekitar pukul 21:00, dan kemudian menurun setelahnya. Ini mungkin disebabkan oleh dimulainya kembali aktivitas industri atau peningkatan lalu lintas pada jam-jam pulang kerja.
    - Konsentrasi PM2.5 dan PM10 juga menunjukkan peningkatan setelah pukul 18:00, yang mungkin terkait dengan emisi pada malam hari dan pengurangan polutan akibat kecepatan angin yang lebih rendah atau lapisan inversi pada malam hari.

    **Periode Peralihan (Sebelum dan Sesudah Jam Kerja):**
    - **Peralihan Pagi:** Terlihat adanya kenaikan konsentrasi PM2.5, NO2, dan SO2 pada periode ini, yang kemungkinan besar disebabkan oleh peningkatan lalu lintas saat jam sibuk pagi dan dimulainya operasi industri.
    - **Peralihan Sore:** Pada periode ini, konsentrasi CO mengalami peningkatan signifikan, sementara polutan lain seperti NO2 dan PM10 juga menunjukkan kenaikan, mencerminkan dampak lalu lintas sore hari dan aktivitas industri setelah jam kerja.

    **Berdasarkan tren yang terlihat setiap jam:**
    - **Polusi Tinggi di Pagi Hari:** Terjadi antara pukul 7:00 - 9:00, terutama untuk polutan PM2.5, PM10, NO2, dan SO2.
    - **Polusi Tinggi di Malam Hari:** Terjadi antara pukul 18:00 - 21:00, terutama untuk polutan CO, PM10, dan O3.
    - **Periode polusi tinggi ini bertepatan dengan periode peralihan dari dan menuju jam kerja, di mana aktivitas kendaraan bermotor dan emisi industri mencapai puncaknya.**

    Dapat disimpulkan bahwa waktu dengan tingkat polusi udara tertinggi di Beijing terjadi pada pagi hari (07:00-09:00) dan malam hari (18:00-21:00), yang merupakan periode peralihan jam kerja. Ini menunjukkan bahwa penerapan langkah-langkah pengendalian kualitas udara pada jam-jam kritis tersebut bisa sangat efektif. Contohnya, regulasi emisi kendaraan bermotor dan pembatasan operasi industri selama waktu-waktu puncak dapat membantu mengurangi tingkat polusi udara secara signifikan.
    """)

    st.header("Heatmap Konsentrasi Polutan Berdasarkan Jam di Semua Stasiun")
    fig, axes = plt.subplots(2, 3, figsize=(14, 14))
    fig.suptitle("Heatmap Konsentrasi Polutan Berdasarkan Jam", fontsize=14)

    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    for i, pollutant in enumerate(pollutants):
        ax = axes[i // 3, i % 3]
        sns.heatmap(hourly_avg.set_index('hour')[pollutant].to_frame(), annot=True, cmap='coolwarm', cbar_kws={'label': 'Konsentrasi Polutan'}, ax=ax)
        ax.set_title(f"Rata-rata {pollutant}")
        ax.set_ylabel("Jam")
        ax.set_xlabel("Konsentrasi")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    st.pyplot(fig)

    st.markdown("""
    **Polusi Partikulat (PM2.5 dan PM10):**
    - Konsentrasi PM2.5 dan PM10 lebih tinggi pada malam hari, mengindikasikan bahwa polusi partikulat dihasilkan dari aktivitas lalu lintas siang hari yang tersisa. Ini juga menunjukkan kurangnya ventilasi alami untuk kota yang menyebabkan akumulasi polutan di udara saat malam hari.

    **Polusi Gas (NO2, CO, SO2):**
    - Konsentrasi NO2 epertinya terkait dengan lalu lintas kendaraan. Hal ini menunjukkan bahwa emisi kendaraan menjadi penyumbang utama polusi udara.
    - Konsentrasi CO juga relatif tinggi sepanjang hari, sumber utama emisi CO berasal dari kendaraan bermotor, meskipun ada penurunan di malam hari.
    - SO2 menunjukkan peningkatan pada siang hari, aktivitas industri atau pembakaran bahan bakar terjadi pada siang hari.

    **Ozon (O3):**
    - Konsentrasi ozon mencapai puncak pada siang hari. Ini menunjukkan bahwa penanganan emisi O3 sangat penting untuk mengurangi tingkat ozon di Beijing.
    """)

st.sidebar.markdown('<div class="sidebar-footer">📊 Dashboard Kualitas Udara Beijing</div>', unsafe_allow_html=True)
st.sidebar.write("Data Analysis Project | mchelle.angelina © Dicoding Indonesia 2024")