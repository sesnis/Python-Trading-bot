# Vajadzīgie importi
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime
from dateutil.relativedelta import relativedelta
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging 
import asyncio
from db import connect_db, insert_data

connect_db()

# Ieslēdz LOGGING - options, DEBUG,INFO, WARNING
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpaca Trading platforma
API_KEY = 'PK3X4HJFF86CNNXLJ0AA'
SECRET_KEY = 'MwVDtALh8uh7DwcD23P3H1H0bzEQo8ePDuZIOrxS'
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)


# Nemainīgie
pairis = 'BTC/USD'
daudzums = 2
# Trading modelis un gaidīšanas laiks
waitTime = 600
data = 0
pozicija_tagat, cena_tagat = 0, 0
prognozeta_cena = 0

async def main():
    '''
    Galvenā funkcija lai dabūtu datus un pārbaudītu trade conditions
    '''

    while True:
        logger.info('----------------------------------------------------')
        pred = stockPred()
        global prognozeta_cena
        prognozeta_cena = pred.predictModel()
        logger.info("Prognozētā cena {0}".format(prognozeta_cena)) # sito

        global prog_cena
        prog_cena = logger.info("Prognozētā cena {0}".format(prognozeta_cena))

        l1 = loop.create_task(check_condition())
        await asyncio.wait([l1])
        await asyncio.sleep(waitTime)


class stockPred:
    def __init__(self,
                 past_days: int = 50,
                 trading_pair: str = 'BTCUSD',
                 exchange: str = 'FTXU',
                 feature: str = 'close',
                 look_back: int = 100,
                 neurons: int = 20,
                 activ_func: str = 'linear',
                 dropout: float = 0.2,
                 loss: str = 'mse',
                 optimizer: str = 'adam',
                 epochs: int = 10,
                 batch_size: int = 20,
                 output_size: int = 1
                 ):
        self.exchange = exchange
        self.feature = feature
        self.look_back = look_back
        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size

    def getAllData(self):

        # Alpaca Market Data Client
        data_client = CryptoHistoricalDataClient()

        time_diff = datetime.now() - relativedelta(hours=1500)

        logger.info("Getting bar data for {0} starting from {1}".format(
            pairis, time_diff))
        # Defining Bar data request parameters
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[pairis],
            timeframe=TimeFrame.Hour,
            start=time_diff
        )
        # Dabū bar data no Alpaca
        df = data_client.get_crypto_bars(request_params).df
        global cena_tagat
        cena_tagat = df.iloc[-1]['close']
        return df

    def getFeature(self, df):
        data = df.filter([self.feature])
        data = data.values
        return data

    def scaleData(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def getTrainData(self, scaled_data):

        # Trenējas ar visiem pieējamajiem datiem (trenējas + tests no DEV)
        x, y = [], []
        for price in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[price - self.look_back:price, :])
            y.append(scaled_data[price, :])
        return np.array(x), np.array(y)

    def LSTM_model(self, input_data):

        # LSTM modeļa parametri un kārtas
        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(
            input_data.shape[1], input_data.shape[2]), return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=self.output_size))
        model.add(Activation(self.activ_func))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def trainModel(self, x, y):
        # Trenēšanās modeļa parametri
        x_train = x[: len(x) - 1]
        y_train = y[: len(x) - 1]
        model = self.LSTM_model(x_train)
        modelfit = model.fit(x_train, y_train, epochs=self.epochs,
                             batch_size=self.batch_size, verbose=1, shuffle=True)
        return model, modelfit

    def predictModel(self):

        # Dabū visus datus
        logger.info("Dabū BTC bar datus")
        df = self.getAllData()

        # Dabū feature (slēģanās cenu)
        logger.info("Dabū Feature: {}".format(self.feature))
        data = self.getFeature(df)

        # scale datus un dabū scaler
        logger.info("Scaling Data")
        scaled_data, scaler = self.scaleData(data)

        # Dabū trenēšanās datus
        logger.info("Dabū trenēšanās datus")
        x_train, y_train = self.getTrainData(scaled_data)

        # Uztaisa un atgriež trenēšanās modeli
        logger.info("Trenēšanās modelis")
        model = self.trainModel(x_train, y_train)[0]

        # Dabū slēgšanās cenu priekš prognozēšanas
        logger.info("Dabū prognozes datus") 
        x_pred = scaled_data[-self.look_back:]
        x_pred = np.reshape(x_pred, (1, x_pred.shape[0]))

        # Paredz rezūltātu
        logger.info("Prognozē cenu")
        pred = model.predict(x_pred).squeeze()
        pred = np.array([float(pred)])
        pred = np.reshape(pred, (pred.shape[0], 1))

        # Inverse scaling lai dabūtu īsto cenu
        pred_true = scaler.inverse_transform(pred)

        return pred_true[0][0]


async def check_condition():
    '''
    Stratēģija:
    - Ja prognozētā cena pēc stundas no šī brīža ir virs pašreizējai cenai un nav pieējama pozīcija, pērk
    - Ja prognozētā cena stundu no šī brīža ir zem pašreizējās cenas un ir pieējama pozīcija, pārdod
    '''
    global pozicija_tagat, cena_tagat, prognozeta_cena
    pozicija_tagat = get_positions()
    logger.info("Current Price is: {0}".format(cena_tagat)) # sito

    global curr_cena
    curr_cena = logger.info("Current Price is: {0}".format(cena_tagat))

    logger.info("Current Position is: {0}".format(pozicija_tagat)) # sito

    global curr_pos
    curr_pos = logger.info("Current Position is: {0}".format(pozicija_tagat))

    # Ja nav pieējama pozīcija un pašreizējā cena ir mazāka nekā prognozētā cena - sūta pirkšanas pasūtījumu
    if float(pozicija_tagat) <= 0.01 and cena_tagat < prognozeta_cena:
        logger.info("Placing Buy Order")
        buy_order = await post_alpaca_order('buy')
        if buy_order:
            logger.info("Buy Order Placed") # sito
            global action1
            action1 = logger.info("Buy Order Placed")
        

    # Ja ir pieējama pozīcija un pašreizējā cena ir lielāka nekā prognozētā cena - sūta pārdošanas pasūtījumu
    if float(pozicija_tagat) >= 0.01 and cena_tagat > prognozeta_cena:
        logger.info("Placing Sell Order")
        sell_order = await post_alpaca_order('sell')
        if sell_order:
            logger.info("Sell Order Placed") # sito
            global action2
            action2 = logger.info("Sell Order Placed")


async def post_alpaca_order(side):
    '''
    Sūta pasūtījumu uz Alpaca
    '''
    try:
        if side == 'buy':
            market_order_data = MarketOrderRequest(
                symbol="BTCUSD",
                qty=daudzums,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            buy_order = trading_client.submit_order(
                order_data=market_order_data
            )

            action = action1
            insertDataInDb(action)
            return buy_order
        else:
            market_order_data = MarketOrderRequest(
                symbol="BTCUSD",
                qty=pozicija_tagat,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            sell_order = trading_client.submit_order(
                order_data=market_order_data
            )

            action = action2
            insertDataInDb(action)
            return sell_order
    except Exception as e:
        logger.exception(
            "Nevarēja izpildīt sūtījumu: {0}".format(e)) # sito        
        return False

def insertDataInDb(action):
    currtime = datetime.now()
    
    data = {
        "prog_cena" : prog_cena,
        "curr_cena" : curr_cena,
        "curr_pos" : curr_pos,
        "action" : action,
        "currtime" : currtime
    }
    print(data)

    insert_data(data)

def get_positions():
    positions = trading_client.get_all_positions()
    global pozicija_tagat
    for p in positions:
        if p.symbol == 'BTCUSD':
            pozicija_tagat = p.qty
            return pozicija_tagat
    return pozicija_tagat

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()