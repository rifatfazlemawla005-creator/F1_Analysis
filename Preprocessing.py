import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, path, target_column, log_dir='logs'):
        """
        Inisialisasi Class Preprocessing Tabular
        :param path: Jalur file CSV data mentah
        :param target_column: Nama kolom yang menjadi target/label (y)
        :param log_dir: Posisi folder untuk menyimpan file log
        """
        self.path = path
        self.target_column = target_column
        self.log_dir = log_dir
        
        # 1. Setup Logging Dinamis
        self._setup_logger()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 2. Ingest Data
        self.logger.info(f"Membaca dataset dari: {self.path}")
        import pandas as pd
        self.data = pd.read_csv(self.path)
        
        # 3. Pisahkan X dan y secara internal berdasarkan target_column
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        self.numerical_features = None
        self.categorical_features = None

    def _setup_logger(self):
        """Membuat folder log dan mengonfigurasi format logging global"""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, 'preprocessing.log')
        
        # Reset handlers jika sudah ada sebelumnya agar tidak duplikat di Jupyter
        logging.root.handlers = []
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def get_clean_pipeline(self):
        """
        Mendeteksi tipe fitur dan membangun ColumnTransformer Pipeline
        :return: Objek ColumnTransformer yang siap dimasukkan ke model utama
        """
        self.logger.info("Mengidentifikasi tipe fitur numerik dan kategorikal...")
        
        # Deteksi otomatis kolom berdasarkan tipe data dari self.X
        self.numerical_features = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        
        self.logger.info(f"Fitur Numerik terdeteksi: {self.numerical_features}")
        self.logger.info(f"Fitur Kategorikal terdeteksi: {self.categorical_features}")

        # Sub-pipeline untuk data angka
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        # Sub-pipeline untuk data teks/kategori
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Menggabungkan kedua sub-pipeline menggunakan ColumnTransformer
        self.logger.info("Menggabungkan seluruh komponen ke dalam ColumnTransformer...")
        preprocessor = ColumnTransformer([
            ('num_pipeline', numerical_pipeline, self.numerical_features),
            ('cat_pipeline', categorical_pipeline, self.categorical_features)
        ])
        
        return preprocessor
        
    def split_data(self, test_size=0.3, random_state=42):
        """Fungsi helper untuk split data langsung dari dalam class"""
        self.logger.info(f"Membagi data dengan test_size={test_size}")
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)