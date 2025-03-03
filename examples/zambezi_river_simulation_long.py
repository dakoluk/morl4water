
import numpy as np
import pandas as pd
import os

class MyFile:
    """
    A class to represent a matrix to-be-read from an external file.

    Attributes
    ----------
    file_name : str
        Path to the data file
    row : int
        Specified number of rows of the file
    col : int
        Specified number of columns of the file
    """

    def __init__(self):
        self.file_name = None
        self.row = None
        self.col = None


class utils:
    """
    A class that provides the static methods which
    are meant to be used by objects from multiple classes.
    """

    @staticmethod
    def loadVector(file_name, l):
        """Loads the vector as a np.array with length=l from the specified
        file. In case the specified path does not exist, raises a print
        statement.

        Parameters
        ----------
        file_name : str
            Path to the file to-be-read
        l : int
            Length of the expected array

        Returns
        -------
        output : np.array
            The array as read from the specified external file. Elements are
            of 'float' type 
        """

        output = np.empty(0)
        try:
            with open(file_name, "r") as input:
                try:
                    count = 0
                    for line in input:
                        for word in line.split():
                            output = np.append(output, float(word))
                            count += 1
                            if (count == l):
                                break
                except:
                    print("Unable to iterate")
        except:
            print("Unable to open file", " ", file_name)

        return output

    @staticmethod
    def loadIntVector(file_name, l):
        """Loads the vector as a np.array with length=l from the specified
        file (elements casted as integer). In case the specified path does
        not exist, raises a print statement.

        Parameters
        ----------
        file_name : str
            Path to the file to-be-read
        l : int
            Length of the expected array

        Returns
        -------
        output : np.array
            The array as read from the specified external file. Elements are
            of 'int' type 
        """

        output = np.empty(0)
        try:
            with open(file_name, "r") as input:
                try:
                    for i, line in enumerate(input):
                        output = np.append(output, int(float(line.strip())))
                        if (i == l - 1):
                            break
                except:
                    print("Unable to iterate")
        except:
            print("Unable to open file" + "   " + file_name)

        return output

    @staticmethod
    def loadMatrix(file_name, row, col):
        """Loads the matrix as a np.array with vertical length=row,
        horizontal length=col from the specified file. In case the
        specified path does not exist, raises a print statement.

        Parameters
        ----------
        file_name : str
            Path to the file to-be-read
        row : int
            Number of rows of the matrix to read
        col : int
            Number of columns of the matrix to read

        Returns
        -------
        output : np.array
            The matrix as read from the specified external file. Elements are
            of 'float' type 
        """

        output = np.empty((row, col))
        try:
            with open(file_name, "r") as input:
                for i, line in enumerate(input):
                    templist = line.split()
                    for j in range(col):
                        output[i][j] = float(templist[j])
        except:
            print("Unable to open file" + "   " + file_name)

        return output

    @staticmethod
    def interp_lin(X, Y, x):
        """Takes two vectors and a number. Based on the relative position
        of the number in the first vector, returns a number that has the
        equivalent relative position in the second vector.

        Parameters
        ----------
        X : np.array
            The array/vector that defines the axis for the input
        Y : np.array
            The array/vector that defines the axis for the output
        x : float
            The input for which an extrapolation is seeked

        Returns
        -------
        y : float
            Extrapolated output 
        """
        dim = X.size - 1

        # extreme cases (x<X(0) or x>X(end): extrapolation
        if (x <= X[0]):
            y = (Y[1] - Y[0]) / (X[1] - X[0]) * (x - X[0]) + Y[0]
            return y

        if (x >= X[dim]):
            y = Y[dim] + (Y[dim] - Y[dim - 1]) / (X[dim] - X[dim - 1]) * \
                (x - X[dim])
            return y

        # otherwise
        # [ x - X(A) ] / [ X(B) - x ] = [ y - Y(A) ] / [ Y(B) - y ]
        # y = [ Y(B)*x - X(A)*Y(B) + X(B)*Y(A) - x*Y(A) ] / [ X(B) - X(A) ]
        delta = 0.0
        min_d = 100000000000.0
        j = -99

        for i in range(X.size):
            if (X[i] == x):
                y = Y[i]
                return y

            delta = abs(X[i] - x)
            if (delta < min_d):
                min_d = delta
                j = i

        k = int()
        if (X[j] < x):
            k = j
        else:
            k = j - 1

        a = (Y[k + 1] - Y[k]) / (X[k + 1] - X[k])
        b = Y[k] - a * X[k]
        y = a * x + b

        return y

    @staticmethod
    def normalizeVector(X, m, M):
        """Normalize an input vector (X) between a minimum (m) and
        maximum (m) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be normalized
        m : np.array
            The array/vector that gives the minimum values
        M : np.array
            The array/vector that gives the maximum values

        Returns
        -------
        Y : np.array
            Normalized vector output 
        """

        Y = np.empty(0)
        for i in range(X.size):
            z = (X[i] - m[i]) / (M[i] - m[i])
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def deNormalizeVector(X, m, M):
        """Retrieves a normalized vector back with respect to a minimum (m) and
        maximum (m) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be denormalized
        m : np.array
            The array/vector that gives the minimum values
        M : np.array
            The array/vector that gives the maximum values

        Returns
        -------
        Y : np.array
            deNormalized vector output 
        """

        Y = np.empty(0)
        for i in range(X.size):
            z = X[i] * (M[i] - m[i]) + m[i]
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def standardizeVector(X, m, s):
        """Standardize an input vector (X) with a minimum (m) and
        standard (s) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be standardized
        m : np.array
            The array/vector that gives the minimum values
        s : np.array
            The array/vector that gives the standard values

        Returns
        -------
        Y : np.array
            Standardized vector output 
        """

        Y = np.empty(0)
        for i in range(X.size):
            z = (X[i] - m[i]) / (s[i])
            Y = np.append(Y, z)

        return Y

    @staticmethod
    def deStandardizeVector(X, m, s):
        """Retrieve back a vector that was standardized with respect to
        a minimum (m) and standard (s) value given per element.

        Parameters
        ----------
        X : np.array
            The array/vector to be destandardized
        m : np.array
            The array/vector that gives the minimum values
        s : np.array
            The array/vector that gives the standard values

        Returns
        -------
        Y : np.array
            deStandardized vector output 
        """
        Y = np.empty(0)
        for i in range(X.size):
            z = X[i] * s[i] + m[i]
            Y = np.append(Y, z)

        return Y



class ReservoirParam:
    """ ReservoirParam class initializes the attributes of the reservoirs
        ReservoirParam : miscellaneous
        Parameters:
    """
    def __init__(self):
        self.EV = int()
        self.evap_rates = MyFile()
        self.rating_curve = MyFile()
        self.rating_curve_minmax = MyFile()
        self.rule_curve = MyFile()
        self.lsv_rel = MyFile()
        self.A = float()  # reservoir surface (assumed to be constant)
        self.initCond = float()
        self.tailwater = MyFile()
        self.minEnvFlow = MyFile()


class Reservoir:
    """
    The reservoir class has functions that contain calculations of the catchments, using storage (s), level (h),
    decision (u), release (r), inflow (q) and surface (S)

    Attributes
    ----------
    ReservoirName : str
        lowercase non-spaced name of the reservoir


    Methods
    -------
    storage_to_level(h=float)
        Returns the level(height) based on volume
    level_to_storage(s=float)
        Returns the volume based on level(height)
    level_to_surface(h=float)
        Returns the surface area based on level
    """

    def __init__(self, name):
        self.ReservoirName = name  # assigning the name of the reservoir in constructor

    def storage_to_level(self, s):
        # interpolation when lsv_rel exists, takes storage (s) and returns level (h)
        if (self.lsv_rel.size > 0):
            h = utils.interp_lin(self.lsv_rel[2], self.lsv_rel[0], s)
        # approximating with volume and cross-section
        else:
            h = s / self.A
        return h

    def level_to_storage(self, h):
        # interpolation when lsv_rel exists
        if (self.lsv_rel.size > 0):
            s = utils.interp_lin(self.lsv_rel[0], self.lsv_rel[2], h)
        # approximating with level and cross section
        else:
            s = h * self.A
        return s

    def level_to_surface(self, h):
        # interpolation when lsv_rel exists. Takes level (h) and returns surface (S)
        if (self.lsv_rel.size > 0):
            S = utils.interp_lin(self.lsv_rel[0], self.lsv_rel[1], h)
        # approximating with cross section
        else:
            S = self.A
        return S

    def min_release(self, s):
        if (self.ReservoirName == "kafuegorgelower"):
            q = 0.0
            if (s <= self.rating_curve_minmax[0]): # rating curve = water level to flow
                q = 0
            elif (s >= self.rating_curve_minmax[1]):
                q = self.rating_curve_minmax[2]
            else:
                q = 0
            return q

        else:
            # no time-dependent (cmonth=0 not used)
            q = 0.0
            if (self.rating_curve.size > 0):
                q = utils.interp_lin(self.rating_curve[0], self.rating_curve[1], self.storage_to_level(s))
            else:
                print(self.ReservoirName, " rating curve not defined")
            return q

    def max_release(self, s):
        if (self.ReservoirName == "kafuegorgelower"):
            q = 0.0
            if (s <= self.rating_curve_minmax[0]):
                q = 0
            elif (s >= self.rating_curve_minmax[1]):
                q = self.rating_curve_minmax[2]
            else:
                q = self.rating_curve_minmax[2]
            return q
        else:
            q = 0.0
            if (self.rating_curve.size > 0):
                q = utils.interp_lin(self.rating_curve[0], self.rating_curve[2], self.storage_to_level(s))
            else:
                print(self.ReservoirName, " rating curve not defined")
            return q

    def actual_release_MEF(self, uu, s, cmonth, n_sim, MEF):

        if (self.ReservoirName == "itezhitezhi"):
            # min-Max storage-discharge relationship
            qm = self.min_release(s)
            qM = self.max_release(s)

            # actual release
            rr = min(qM, max(qm, uu))

            # compute actual release - NO FLOW AUGMENTATION for Itezhitezhi
            rr_MEF = 0.0

            if (MEF <= 40):  # Itezhitezhi MUST release a MF of 40 m3/sec all year round
                rr_MEF = max(rr, MEF)
            else:
                if (n_sim <= 40):
                    rr_MEF = max(rr, 40)
                elif (n_sim > 40 and n_sim < MEF):
                    rr_MEF = max(rr, n_sim)
                elif (n_sim >= MEF):
                    rr_MEF = max(rr, MEF)

            return rr_MEF

        else:
            # min-Max storage-discharge relationship
            qm = self.min_release(s)
            qM = self.max_release(s)

            # actual release
            rr = min(qM, max(qm, uu))
            rr_MEF = max(rr, MEF)

            return rr_MEF

    def integration(self, HH, tt, s0, uu, n_sim, cmonth):  # returns double vector!
        # HH = number of days in the current month
        # tt = current month

        self.sim_step = 3600 * 2 * HH / HH
        HH = int(HH)

        self.s = np.full(HH + 1, -999.0).astype('float')
        self.r = np.full(HH, -999)
        self.stor_rel = np.empty(0)

        self.MEF = self.getMEF(cmonth - 1)

        # initial conditions
        self.s[0] = s0

        for i in range(HH):
            # compute actual release - NO FLOW AUGMENTATION
            self.r[i] = self.actual_release_MEF(uu, self.s[i], cmonth, n_sim, self.MEF)

            # compute evaporation
            if (self.EV == 1):
                self.S = self.level_to_surface(self.storage_to_level(self.s[i]))
                self.E = self.evap_rates[cmonth - 1] / 1000 * self.S / (3600 * 2 * HH)
            #  One elif to be implemented (?)
            else:
                self.E = 0.0

            # system transition
            self.s[i + 1] = self.s[i] + self.sim_step * (n_sim - self.r[i] - self.E)

        self.stor_rel = np.append(self.stor_rel, self.s[HH])
        self.stor_rel = np.append(self.stor_rel, np.mean(self.r))

        return self.stor_rel

    def integration_daily(self, HH, tt, s0, uu, n_sim, cmonth):
        # HH = number of days in the current month
        # tt = current month

        self.sim_step = 3600 * 24 * HH / HH

        HH = int(HH)

        self.s = np.full(HH + 1, -999, dtype=float)

        self.r = np.full(HH, -999)
        self.stor_rel = np.empty(0)

        self.MEF = self.getMEF(cmonth - 1)

        # initial conditions
        self.s[0] = s0

        for i in range(HH):
            # compute actual release - NO FLOW AUGMENTATION
            self.r[i] = self.actual_release_MEF(uu, self.s[i], cmonth, n_sim, self.MEF)

            # compute evaporation
            if (self.EV == 1):
                self.S = self.level_to_surface(self.storage_to_level(self.s[i]))
                self.E = self.evap_rates[cmonth - 1] / 1000 * self.S / (86400 * HH)
            # One elif to be implemented!!
            else:
                self.E = 0.0

            # system transition
            self.s[i + 1] = self.s[i] + self.sim_step * (n_sim - self.r[i] - self.E)

        self.stor_rel = np.append(self.stor_rel, self.s[HH])
        self.stor_rel = np.append(self.stor_rel, np.mean(self.r))

        return self.stor_rel

    def actual_release(self, uu, s, cmonth):
        # min-Max storage-discharge relationship
        qm = self.min_release(s)
        qM = self.max_release(s)

        # actual release
        rr = min(qM, max(qm, uu))

        return rr

    def relToTailwater(self, r):
        hd = 0.0
        if (self.tailwater.size > 0):
            hd = utils.interp_lin(self.tailwater[0], self.tailwater[1], r)

        return hd

    def setInitCond(self, ci):
        self.init_condition = ci

    def getInitCond(self):
        return self.init_condition

    def setEvap(self, pEV):
        self.EV = pEV

    def setEvapRates(self, pEvap):
        self.evap_rates = utils.loadVector(pEvap.file_name, pEvap.row)

    def setRatCurve(self, pRatCurve):
        self.rating_curve = utils.loadMatrix(pRatCurve.file_name, pRatCurve.row, pRatCurve.col)

    def setRatCurve_MinMax(self, pRatCurve_MinMax):
        self.rating_curve_minmax = utils.loadVector(pRatCurve_MinMax.file_name, pRatCurve_MinMax.col)

    def setRuleCurve(self, pRuleCurve):
        self.rule_curve = utils.loadMatrix(pRuleCurve.file_name, pRuleCurve.row, pRuleCurve.col)

    def setLSV_Rel(self, pLSV_Rel):
        self.lsv_rel = utils.loadMatrix(pLSV_Rel.file_name, pLSV_Rel.row, pLSV_Rel.col)

    def setSurface(self, pA):
        self.A = pA

    def setTailwater(self, pTailWater):
        self.tailwater = utils.loadMatrix(pTailWater.file_name, pTailWater.row, pTailWater.col)

    def setMEF(self, pMEF):
        self.minEnvFlow = utils.loadVector(pMEF.file_name, pMEF.row)

    def getMEF(self, pMoy):
        return self.minEnvFlow[pMoy]


class CatchmentParam:
    # The CatchmentParam class defines CM
    def __init__(self) -> None:
        self.CM = int(0)  # type of catchment model_redriver (0 = historical trajectory, 1 = HBV)
        # HBV = Hydrologiska Byråns Vattenbalansavdelning model. A hydrological transport model to measure the
        # streamflow (Arnold et. al, 2023). However, this is not implemented in the code (neither in c++ code).
        self.inflow_file = MyFile()  # myFile type from utils. Contains the inflow trajectory


class Catchment:
    # loads the inflow of the catchment.
    def __init__(self, pCM):
        cModel = pCM.CM
        self.inflow = utils.loadVector(pCM.inflow_file.file_name, pCM.inflow_file.row) # overwritten in

        if cModel == 0:
            self.inflow = utils.loadVector(pCM.inflow_file.file_name, pCM.inflow_file.row)

    def get_inflow(self, pt):
        """ function to retrieve the inflow for day "pt" """
        q = float(self.inflow[pt])
        return q



class Policy:
    def __init__(self):
        self.functions = dict()
        self.approximator_names = list()
        self.all_parameters = np.empty(0)

    def add_policy_function(self, name, type, n_inputs, n_outputs, **kwargs):
        """
        Adds the policy functions to the policy class, based on the "type" given as input.
        The policy functions are either RBF, user specified or ANN. ANN are not configured currently.
        The RBF function is defined in the ncRBF method and the user specified
        function is defined in the klass method.

        Placeholder < ?
        """

        if type == "ncRBF":
            self.functions[name] = ncRBF(n_inputs, n_outputs, kwargs['n_structures'])  # name = release

        elif type == "user_specified":
            class_name = kwargs['class_name']  # name = irrigation, class_name = IrrigationPolicy
            klass = globals()[class_name]
            self.functions[name] = klass(n_inputs, n_outputs, kwargs)

        elif type == "ANN":
            pass

        # Create a list of all the policy names (release, irrigation)
        self.approximator_names.append(name)
        # self.all_parameters = np.append

    def assign_free_parameters(self, full_array):
        """
        The assign_free_parameters functions uses the get_free_parameter_number() function to calculate the # of parameters
        for the irrigation policies (in this smash module) and for the release policies (in alternative_policy_
        structures).
        The function is used to assign all policy parameters (free parameter = not predefined parameter) to the policy class

        """
        beginning = 0
        for name in self.approximator_names:
            end = beginning + self.functions[name].get_free_parameter_number()
            self.functions[name].set_parameters(full_array[beginning:end])
            beginning = end






class ModelZambezi:
    """
    Model class consists of three major functions. First, static components
    such as reservoirs, catchments, policy objects are created within the
    constructor. Evaluate function serves as the behaviour generating machine
    which outputs KPIs of the model. Evaluate does so by means of calling
    the simulate function which handles the state transformation via
    mass-balance equation calculations iteratively.
    """

    def __init__(self, policy_sim):
        """
        Creating the static objects of the model including the reservoirs,
        catchments and policy objects along with their parameters. Also,
        reading both the model run configuration from settings,
        input data (flows etc.) as well as policy function hyper-parameters.
        """

        # initialize parameter constructs for objects (policy and Catchment)
        # (has to be done before file reading)

        # initialize parameter constructs for to be created policy objects   
        self.p_param = policy_parameters_construct()
        self.irr_param = irr_function_parameters()

        # initialize parameter constructs for to be created
        # Catchment parameter objects (stored in a dictionary):
        catchment_list = ["Itt", "KafueFlats", "Ka", "Cb", "Cuando", "Shire", "Bg"]
        self.catchment_param_dict = dict()

        for catchment_name in catchment_list:
            catch_param_name = catchment_name + "_catch_param"
            self.catchment_param_dict[catch_param_name] = CatchmentParam()

        # Reservoir parameter objects (stored seperately
        # to facilitate settings file reference):
        self.KGU_param = ReservoirParam()
        self.ITT_param = ReservoirParam()
        self.KA_param = ReservoirParam()
        self.CB_param = ReservoirParam()
        self.KGL_param = ReservoirParam()

        # read the parameter values from either CSV or UI
        self.readFileSettings()

        # Catchment objects (stored in a dictionary)
        self.catchment_dict = dict()

        for catchment_name in catchment_list:
            catch_param_name = catchment_name + "_catch_param"
            variable_name = catchment_name + "Catchment"

            # Specific parameter construct is used in instantiation
            self.catchment_dict[variable_name] = Catchment(self.catchment_param_dict[catch_param_name])

        ###################
        # CREATE RESERVOIRS
        ###################
        # each of the 5 existing reservoirs is created here

        # 1. KAFUE GORGE UPPER (KGU) reservoir
        self.KafueGorgeUpper = Reservoir("kafuegorgeupper")  # creating a new object from corresponding Reservoir class
        # 2. ITEZHITEZHI (ITT) reservoir
        self.Itezhitezhi = Reservoir("itezhitezhi")
        # 3. KARIBA (KA) reservoir
        self.Kariba = Reservoir("kariba")
        # 4. CAHORA BASSA (CB) reservoir
        self.CahoraBassa = Reservoir("cahorabassa")
        # 5. KAFUE GORGE LOWER reservoir
        self.KafueGorgeLower = Reservoir("kafuegorgelower")

        # Add reservoir evaporation rates
        # 1 KGU
        self.KafueGorgeUpper.setEvap(
            1)  # evaporation data: 0 = no evaporation, 1 = load evaporation from file, 2 = activate function
        self.KGU_param.evap_rates.file_name = "../data/evap_KG_KF.txt"
        self.KGU_param.evap_rates.row = self.T  # setting # of rows as simulation period (12) for file reading
        self.KafueGorgeUpper.setEvapRates(self.KGU_param.evap_rates)
        # 2 ITT
        self.Itezhitezhi.setEvap(1)
        self.ITT_param.evap_rates.file_name = "../data/evap_ITT.txt"
        self.ITT_param.evap_rates.row = self.T
        self.Itezhitezhi.setEvapRates(self.ITT_param.evap_rates)
        # 3 KA
        self.Kariba.setEvap(1)
        self.KA_param.evap_rates.file_name = "../data/evap_KA.txt"
        self.KA_param.evap_rates.row = self.T
        self.Kariba.setEvapRates(self.KA_param.evap_rates)
        # 4 CB
        self.CahoraBassa.setEvap(1)
        self.CB_param.evap_rates.file_name = "../data/evap_CB.txt"
        self.CB_param.evap_rates.row = self.T
        self.CahoraBassa.setEvapRates(self.CB_param.evap_rates)
        # 5 KGL
        self.KafueGorgeLower.setEvap(1)
        self.KGL_param.evap_rates.file_name = "../data/evap_KGL.txt"
        self.KGL_param.evap_rates.row = self.T
        self.KafueGorgeLower.setEvapRates(self.KGL_param.evap_rates)

        # Add reservoir level to storage relation
        # 1 KGU
        self.KGU_param.lsv_rel.file_name = "../data/lsv_rel_KafueGorgeUpper.txt"  # [m m2 m3]
        self.KGU_param.lsv_rel.row = 3
        self.KGU_param.lsv_rel.col = 22
        self.KafueGorgeUpper.setLSV_Rel(self.KGU_param.lsv_rel)
        # 2 ITT
        self.ITT_param.lsv_rel.file_name = "../data/lsv_rel_Itezhitezhi.txt"
        self.ITT_param.lsv_rel.row = 3
        self.ITT_param.lsv_rel.col = 19  #
        self.Itezhitezhi.setLSV_Rel(self.ITT_param.lsv_rel)
        # 3 KA
        self.KA_param.lsv_rel.file_name = "../data/lsv_rel_Kariba.txt"
        self.KA_param.lsv_rel.row = 3
        self.KA_param.lsv_rel.col = 16
        self.Kariba.setLSV_Rel(self.KA_param.lsv_rel)
        # 4 CB
        self.CB_param.lsv_rel.file_name = "../data/lsv_rel_CahoraBassa.txt"
        self.CB_param.lsv_rel.row = 3
        self.CB_param.lsv_rel.col = 10  #
        self.CahoraBassa.setLSV_Rel(self.CB_param.lsv_rel)
        # 5 KGL
        self.KGL_param.lsv_rel.file_name = "../data/lsv_rel_KafueGorgeLower.txt"
        self.KGL_param.lsv_rel.row = 3
        self.KGL_param.lsv_rel.col = 10
        self.KafueGorgeLower.setLSV_Rel(self.KGL_param.lsv_rel)

        # Add the rating curve of the reservoirs
        # 1 KGU
        self.KGU_param.rating_curve.file_name = "../data/min_max_release_KafueGorgeUpper.txt"  # [m m3/s m3/s] #
        self.KGU_param.rating_curve.row = 3
        self.KGU_param.rating_curve.col = 18
        self.KafueGorgeUpper.setRatCurve(self.KGU_param.rating_curve)
        # 2 ITT
        self.ITT_param.rating_curve.file_name = "../data/min_max_release_Itezhitezhi.txt"
        self.ITT_param.rating_curve.row = 3
        self.ITT_param.rating_curve.col = 43
        self.Itezhitezhi.setRatCurve(self.ITT_param.rating_curve)
        # 3 KA
        self.KA_param.rating_curve.file_name = "../data/min_max_release_Kariba.txt"
        self.KA_param.rating_curve.row = 3
        self.KA_param.rating_curve.col = 11
        self.Kariba.setRatCurve(self.KA_param.rating_curve)
        # 4 CB
        self.CB_param.rating_curve.file_name = "../data/min_max_release_CahoraBassa.txt"
        self.CB_param.rating_curve.row = 3
        self.CB_param.rating_curve.col = 10
        self.CahoraBassa.setRatCurve(self.CB_param.rating_curve)
        # 5 KGL
        self.KGL_param.rating_curve_minmax.file_name = "../data/min_max_release_KafueGorgeLower.txt"
        self.KGL_param.rating_curve.row = 1
        self.KGL_param.rating_curve.col = 3
        self.KafueGorgeLower.setRatCurve_MinMax(self.KGL_param.rating_curve_minmax)

        # Add the rule curve of the reservoir
        # 3 KA
        self.KA_param.rule_curve.file_name = "../data/rule_curve_Kariba.txt"
        self.KA_param.rule_curve.row = 3
        self.KA_param.rule_curve.col = 12
        self.Kariba.setRuleCurve(self.KA_param.rule_curve)
        # 4
        self.CB_param.rule_curve.file_name = "../data/rule_curve_CahoraBassa.txt"  # [time(month) month-end level(m) month-end storage(m3)]
        self.CB_param.rule_curve.row = 3
        self.CB_param.rule_curve.col = 12
        self.CahoraBassa.setRuleCurve(self.CB_param.rule_curve)

        # Add the tailwater rating of the reservoirs
        # 1 KGU
        self.KGU_param.tailwater.file_name = "../data/tailwater_rating_KafueGorgeUpper.txt"  # [m3/s m] = [Discharge Tailwater Level]
        self.KGU_param.tailwater.row = 2
        self.KGU_param.tailwater.col = 7
        self.KafueGorgeUpper.setTailwater(self.KGU_param.tailwater)
        # 2 ITT
        self.ITT_param.tailwater.file_name = "../data/tailwater_rating_Itezhitezhi.txt"
        self.ITT_param.tailwater.row = 2
        self.ITT_param.tailwater.col = 10
        self.Itezhitezhi.setTailwater(self.ITT_param.tailwater)
        # 3 KA
        self.KA_param.tailwater.file_name = "../data/tailwater_rating_Kariba.txt"
        self.KA_param.tailwater.row = 2
        self.KA_param.tailwater.col = 9
        self.Kariba.setTailwater(self.KA_param.tailwater)
        # 4 CB
        self.CB_param.tailwater.file_name = "../data/tailwater_rating_CahoraBassa.txt"
        self.CB_param.tailwater.row = 2
        self.CB_param.tailwater.col = 9
        self.CahoraBassa.setTailwater(self.CB_param.tailwater)
        # 5 KBL
        self.KGL_param.tailwater.file_name = "../data/tailwater_rating_KafueGorgeLower.txt"
        self.KGL_param.tailwater.row = 2
        self.KGL_param.tailwater.col = 8
        self.KafueGorgeLower.setTailwater(self.KGL_param.tailwater)

        # Add the minimum environmental flow of the reservoirs
        # 1 KGU
        self.KGU_param.minEnvFlow.file_name = "../data/MEF_KafueGorgeUpper.txt"  # [m^3/sec]
        self.KGU_param.minEnvFlow.row = self.T
        self.KafueGorgeUpper.setMEF(self.KGU_param.minEnvFlow)
        self.KafueGorgeUpper.setInitCond(self.KGU_param.initCond)
        # 2 ITT
        self.ITT_param.minEnvFlow.file_name = "../data/MEF_Itezhitezhi.txt"
        self.ITT_param.minEnvFlow.row = self.T
        self.Itezhitezhi.setMEF(self.ITT_param.minEnvFlow)
        self.Itezhitezhi.setInitCond(self.ITT_param.initCond)
        # 3 KA
        self.KA_param.minEnvFlow.file_name = "../data/MEF_Kariba.txt"
        self.KA_param.minEnvFlow.row = self.T
        self.Kariba.setMEF(self.KA_param.minEnvFlow)
        self.Kariba.setInitCond(self.KA_param.initCond)
        # 4 CB
        self.CB_param.minEnvFlow.file_name = "../data/MEF_CahoraBassa.txt"
        self.CB_param.minEnvFlow.row = self.T
        self.CahoraBassa.setMEF(self.CB_param.minEnvFlow)
        self.CahoraBassa.setInitCond(self.CB_param.initCond)
        # 5 KGL
        self.KGL_param.minEnvFlow.file_name = "../data/MEF_KafueGorgeLower.txt"
        self.KGL_param.minEnvFlow.row = self.T
        self.KafueGorgeLower.setMEF(self.KGL_param.minEnvFlow)
        self.KafueGorgeLower.setInitCond(self.KGL_param.initCond)

        # Below the policy objects (from the SMASH library) are generated
        # this model requires two policy functions (to be used in seperate
        # places in the simulate function) which are the "release" and
        # "irrigation" policies. While the former is meant to be a generic
        # approximator such as RBF and ANN (to be optimized) the latter
        # has a simple structure specified in the
        # alternative_policy_structures script. Firstly, a Policy object is
        # instantiated which is meant to own all policy functions within a
        # model (see the documentation of SMASH). Then, two separate policies
        # are added onto the overarching_policy.

        self.overarching_policy = Policy()

        self.overarching_policy.add_policy_function(name="irrigation",
                                                    type="user_specified", n_inputs=4, n_outputs=1,
                                                    class_name="IrrigationPolicy", n_irr_districts=8)

        self.overarching_policy.functions["irrigation"].set_min_input(self.irr_param.mParam)
        self.overarching_policy.functions["irrigation"].set_max_input(self.irr_param.MParam)

        self.overarching_policy.add_policy_function(name="release",
                                                    type="ncRBF", n_inputs=self.p_param.policyInput,
                                                    n_outputs=self.p_param.policyOutput,
                                                    n_structures=self.p_param.policyStr)

        self.overarching_policy.functions["release"].set_max_input(self.p_param.MIn)
        self.overarching_policy.functions["release"].setMaxOutput(self.p_param.MOut)
        self.overarching_policy.functions["release"].set_min_input(self.p_param.mIn)
        self.overarching_policy.functions["release"].setMinOutput(self.p_param.mOut)

        # Load irrigation demand vectors (stored in a dictionary)
        irr_naming_list = range(2, 10, 1)
        self.irr_demand_dict = dict()

        for id in irr_naming_list:
            variable_name = "irr_demand" + str(id)
            file_name = "../data/IrrDemand" + str(id) + ".txt"
            self.irr_demand_dict[variable_name] = utils.loadVector(file_name, self.T)

        self.irr_district_idx = utils.loadVector("../data/IrrDistrict_idx.txt",
                                                 self.irr_param.num_irr)  # index referring to the position
        # of the first parameter (hdg) of
        # each irrigation district in the
        # decision variables vector

        # Target hydropower production for each reservoir
        self.tp_Itt = utils.loadVector("../data/ITTprod.txt", self.T)  # Itezhitezhi target production
        self.tp_Kgu = utils.loadVector("../data/KGUprod.txt", self.T)  # Kafue Gorge Upper target production
        self.tp_Ka = utils.loadVector("../data/KAprod.txt", self.T)  # Kariba target production
        self.tp_Cb = utils.loadVector("../data/CBprod.txt", self.T)  # Cahora Bassa target production
        self.tp_Kgl = utils.loadVector("../data/KGLprod.txt", self.T)  # Kafue Gorge Lower target production

        # Load Minimum Environmental Flow requirement upstream of Victoria Falls
        self.MEF_VictoriaFalls = utils.loadVector("../data/MEF_VictoriaFalls.txt", self.T)  # [m^3/sec]

        # Load Minimum Environmental Flow requirement in the Zambezi Delta for the months of February and March
        self.qDelta = utils.loadVector("../data/MEF_Delta.txt", self.T)  # [m^3/sec]

        ### First difference with model_zambezi_OPT
        self.PolicySim = policy_sim  # To keep the name of the policy with which the simulatin is run
        ### end of difference

    def getNobj(self):
        return self.Nobj

    def getNvar(self):
        return self.Nvar

    def evaluate(self, var):
        """ Evaluate the KPI values based on the given input
        data and policy parameter configuration.

        Parameters
        ----------
        self : ModelZambezi object
        var : np.array
            Parameter values for the reservoir control policy
            object (NN, RBF etc.)

        Returns
        -------
        Either obj or None (just writing to a file) depending on
        the mode (simulation or optimization)
        """

        obj = np.empty(0)

        ### Second difference
        objectives = open("../objs/bc/" + self.PolicySim + "_simulated.objs",
                          'w+')  # opening the file to write the objective values
        ###

        self.overarching_policy.assign_free_parameters(var)

        if (self.Nsim < 2):  # single simulation
            J = self.simulate()
            obj = J

        else:  # MC Simulation to be adjusted
            Jhyd = np.empty()
            Jenv = np.empty()
            Jirr_def = np.empty()

            for _ in range(self.Nsim):
                J = self.simulate
                Jhyd = np.append(Jhyd, J[0])
                Jenv = np.append(Jenv, J[1])
                Jirr_def = np.append(Jirr_def, J[2])

            # objectives aggregation (average of Jhyd + worst 1st percentile for Jenv, Jirr)
            obj = np.append(obj, np.mean(Jhyd))
            obj = np.append(obj, np.percentile(Jenv, 99))
            obj = np.append(obj, np.percentile(Jirr_def, 99))

        ### 3rd difference OPT
        obj_string = ''

        for i in range(len(obj)):
            obj_string += str(obj[i]) + ' '
        print('objectives:',obj_string)
        objectives.write(obj_string)
        #objectives.write(str(obj[0]) + ' ' + str(obj[1]) + ' ' + str(obj[2]))
        #print('objectives:',str(obj[0]) + ' ' + str(obj[1]) + ' ' + str(obj[2]))
        objectives.close()
        ###

        # re-initialize policy parameters for further runs in the optimization mode
        self.overarching_policy.functions["release"].clear_parameters()
        self.overarching_policy.functions["irrigation"].clear_parameters()

    def simulate(self):
        """ Mathematical simulation over the specified simulation
        duration within a main for loop based on the mass-balance
        equation

        Parameters
        ----------
        self : ModelZambezi object
            
        Returns
        -------
        JJ : np.array
            Array of calculated KPI values
        """

        ### Diff 4
        # Opening the files (ofstream in c++)
        folder_path = "../storage_release/bc_policy_simulation/"
        os.makedirs(folder_path, exist_ok=True)

        # Open or create the file for writing
        irrigation = open(os.path.join(folder_path, "irr_" + self.PolicySim + ".txt"), 'w+')
        rDelta = open(os.path.join(folder_path +"rDelta_" + self.PolicySim + ".txt"), 'w+')


        # Initialize the mass_balance
        mass_balance_ReservoirSim = dict()
        qturb_ReservoirSim = dict()
        hydropower_def_ReservoirSim = dict()

        for reservoir in ['cb', 'itt', 'ka', 'kgu', 'kgl']:
            qturb_ReservoirSim[reservoir] = \
                open(os.path.join(folder_path, "qturb_" + reservoir + "_" + self.PolicySim + ".txt"), 'w+')
            mass_balance_ReservoirSim[reservoir] = open(os.path.join(folder_path + reservoir + "_" + self.PolicySim + ".txt"), 'w+')
            hydropower_def_ReservoirSim[reservoir] = open(
                os.path.join(folder_path + reservoir + "_hydDeficit_" + self.PolicySim + ".txt"), 'w+')
        ### End diff

        ## INITIALIZATION: storage (s), level (h), decision (u), release(r) (Hydropower) : np.array
        import numpy as np

        # storage (s)
        s_kgu = np.full(self.H + 1, -999).astype('float')
        s_itt = np.full(self.H + 1, -999).astype('float')
        s_ka = np.full(self.H + 1, -999).astype('float')
        s_cb = np.full(self.H + 1, -999).astype('float')
        s_kgl = np.full(self.H + 1, -999).astype('float')

        # level (h)
        h_kgu = np.full(self.H + 1, -999).astype('float')
        h_itt = np.full(self.H + 1, -999).astype('float')
        h_ka = np.full(self.H + 1, -999).astype('float')
        h_cb = np.full(self.H + 1, -999).astype('float')
        h_kgl = np.full(self.H + 1, -999).astype('float')

        # decision (u)
        u_kgu = np.full(self.H, -999).astype('float')
        u_itt = np.full(self.H, -999).astype('float')
        u_ka = np.full(self.H, -999).astype('float')
        u_cb = np.full(self.H, -999).astype('float')
        u_kgl = np.full(self.H + 1, -999).astype('float')

        # release (r)
        r_kgu = np.full(self.H + 1, -999).astype('float')
        r_itt = np.full(self.H + 1, -999).astype('float')
        r_itt_delay = np.full(self.H + 3, -999).astype(
            'float')  # 2 months delay between Itezhitezhi and Kafue Gorge Upper
        r_ka = np.full(self.H + 1, -999).astype('float')
        r_cb = np.full(self.H + 1, -999).astype('float')
        r_kgl = np.full(self.H + 1, -999).astype('float')

        moy = np.full(self.H, -999, np.int64)  # Month of the year: integer vector (others float!)

        # release (r) at irrigation areas (irr)
        # r_irr1 = np.full(self.H + 1, -999)
        r_irr2 = np.full(self.H + 1, -999)
        r_irr3 = np.full(self.H + 1, -999)
        r_irr4 = np.full(self.H + 1, -999)
        r_irr5 = np.full(self.H + 1, -999)
        r_irr6 = np.full(self.H + 1, -999)
        r_irr7 = np.full(self.H + 1, -999)
        r_irr8 = np.full(self.H + 1, -999)
        r_irr9 = np.full(self.H + 1, -999)

        # simulation variables Python -. (initialized as float of value 0 and empty np array) 
        q_Itt, q_KafueFlats, q_KaLat, q_Bg, q_Cb, q_Cuando, q_Shire, \
            qTurb_Temp, qTurb_Temp_N, qTurb_Temp_S, headTemp, \
            hydTemp, hydTemp_dist, hydTemp_N, hydTemp_S, irrDef_Temp, irrDefNorm_Temp, \
            envDef_Temp, qTotIN, qTotIN_1 = tuple(20 * [float()])  #
        sd_rd = np.empty(0)  # storage and release resulting from daily integration
        uu = np.empty(0)

        ## DIFF 1 cases
        gg_hydKGU, gg_hydITT, gg_hydKA, gg_hydCB, gg_hydKGL, gg_hydVF, \
            deficitHYD_tot, gg_irr2, gg_irr3, gg_irr4, gg_irr5, gg_irr6, gg_irr7, \
            gg_irr8, gg_irr9, gg_irr2_NormDef, gg_irr3_NormDef, gg_irr4_NormDef, \
            gg_irr5_NormDef, gg_irr6_NormDef, gg_irr7_NormDef, gg_irr8_NormDef, \
            gg_irr9_NormDef, deficitIRR_tot, gg_env, deficitENV_tot = tuple(26 * [np.empty(0)])
        input, outputDEF = tuple([np.empty(0), np.empty(0)])

        # initial condition
        s_kgu[0] = self.KafueGorgeUpper.getInitCond()
        s_itt[0] = self.Itezhitezhi.getInitCond()
        s_ka[0] = self.Kariba.getInitCond()
        s_cb[0] = self.CahoraBassa.getInitCond()
        s_kgl[0] = self.KafueGorgeLower.getInitCond()

        qTotIN_1 = self.inflowTOT00  # initial inflow

        r_itt_delay[0] = 0  # September 1985 [m^3/sec] #
        r_itt_delay[
            1] = 56.183290322580640  # October 1985 [m^3/sec] but divided for the number of days in January 1986 when it actually enters KGU #
        r_itt_delay[
            2] = 59.670678571428570  # November 1985 [m^3/sec] but divided for the number of days in February 1986 when it actually enters KGU #
        r_itt_delay[
            3] = 101.7307419354839  # December 1985 [m^3/sec] but divided for the number of days in March 1986 when it actually enters KGU #

        #################
        # RUN SIMULATION
        #################

        for t in range(self.H):
            # month of the year
            moy[t] = (self.initMonth + t - 1) % (self.T) + 1

            # inflows
            q_Itt = self.catchment_dict["IttCatchment"].get_inflow(t)  # Itezhitezhi inflow @ Kafue Hook Bridge #
            q_KafueFlats = self.catchment_dict["KafueFlatsCatchment"].get_inflow(
                t)  # lateral flow @ Kafue Flats (upstream of Kafue Gorge Upper) #
            q_KaLat = self.catchment_dict["KaCatchment"].get_inflow(
                t)  # Kariba inflow @ Victoria Falls increased by +10% #
            q_Cb = self.catchment_dict["CbCatchment"].get_inflow(
                t)  # Cahora Bassa inflow (Luangwa and other tributaries) #
            q_Cuando = self.catchment_dict["CuandoCatchment"].get_inflow(t)  # Kariba inflow @ Cuando river #
            q_Shire = self.catchment_dict["ShireCatchment"].get_inflow(t)  # Shire discharge (upstream of the Delta) #
            q_Bg = self.catchment_dict["BgCatchment"].get_inflow(
                t)  # Kariba inflow @ Victoria Falls increased by +10% #

            qTotIN = q_Itt + q_KafueFlats + q_KaLat + q_Cb + q_Cuando + q_Shire + q_Bg  # total inflows

            # Add the INPUTS for the function approximator (RBF / ANN) black-box policy
            input = np.array([s_itt[t], s_kgu[t], s_ka[t], s_cb[t], s_kgl[t], moy[t], qTotIN_1])

            # call the POLICY function!
            uu = self.overarching_policy.functions["release"].get_norm_output(input)
            # decision per reservoir initialized
            u_itt[t], u_kgu[t], u_ka[t], u_cb[t], u_kgl[t] = tuple(uu)

            # daily integration and assignment of monthly storage and release values
            # ITT
            sd_rd = self.Itezhitezhi.integration(12 * self.integrationStep[moy[t] - 1], t, s_itt[t], u_itt[t], q_Itt,
                                                 moy[t])  # 2 # 24*integrationStep[moy[t]-1] ORIG
            s_itt[t + 1] = sd_rd[0]  # storage
            r_itt[t + 1] = sd_rd[1]  # release
            r_itt_delay[t + 3] = r_itt[t + 1] * (self.integrationStep[moy[t] - 1]) / (
                self.integrationStep_delay[moy[t] - 1])

            # compute the irrigation water diversion volume [m3/s] for irrigation district 4
            r_irr4[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [q_KafueFlats + r_itt_delay[t + 1], self.irr_demand_dict["irr_demand4"][moy[t] - 1], 4,
                 self.irr_district_idx])

            sd_rd = self.KafueGorgeUpper.integration(12 * self.integrationStep[moy[t] - 1], t, s_kgu[t], u_kgu[t],
                                                     q_KafueFlats + r_itt_delay[t + 1] - r_irr4[t + 1],
                                                     moy[t])  # 2 24*integrationStep[moy[t]-1] ORIG
            s_kgu[t + 1] = sd_rd[0]
            r_kgu[t + 1] = sd_rd[1]

            sd_rd = self.KafueGorgeLower.integration(12 * self.integrationStep[moy[t] - 1], t, s_kgl[t], u_kgl[t],
                                                     r_kgu[t + 1], moy[t])  # 2 24*integrationStep[moy[t]-1]
            s_kgl[t + 1] = sd_rd[0]
            r_kgl[t + 1] = sd_rd[1]

            r_irr2[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [q_Bg + q_Cuando + q_KaLat, self.irr_demand_dict["irr_demand2"][moy[t] - 1], 2,
                 self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #

            sd_rd = self.Kariba.integration_daily(self.integrationStep[moy[t] - 1], t, s_ka[t], u_ka[t],
                                                  q_Bg + q_Cuando + q_KaLat - r_irr2[t + 1], moy[t])  # 2
            s_ka[t + 1] = sd_rd[0]
            r_ka[t + 1] = sd_rd[1]

            r_irr3[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [r_ka[t + 1], self.irr_demand_dict["irr_demand3"][moy[t] - 1], 3,
                 self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #
            r_irr5[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [r_kgl[t + 1], self.irr_demand_dict["irr_demand5"][moy[t] - 1], 5,
                 self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #
            r_irr6[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [r_ka[t + 1] - r_irr3[t + 1] + r_kgl[t + 1] - r_irr5[t + 1],
                 self.irr_demand_dict["irr_demand6"][moy[t] - 1], 6,
                 self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #

            sd_rd = self.CahoraBassa.integration(12 * self.integrationStep[moy[t] - 1], t, s_cb[t], u_cb[t],
                                                 q_Cb + r_kgl[t + 1] + r_ka[t + 1] - (
                                                             r_irr3[t + 1] + r_irr5[t + 1] + r_irr6[t + 1]), moy[t])
            s_cb[t + 1] = sd_rd[0]  #
            r_cb[t + 1] = sd_rd[1]  #
            del sd_rd  #

            r_irr7[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [r_cb[t + 1], self.irr_demand_dict["irr_demand7"][moy[t] - 1], 7,
                 self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #
            r_irr8[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [r_cb[t + 1] - r_irr7[t + 1], self.irr_demand_dict["irr_demand8"][moy[t] - 1], 8,
                 self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #
            r_irr9[t + 1] = self.overarching_policy.functions["irrigation"].get_output(
                [r_cb[t + 1] - r_irr7[t + 1] + q_Shire - r_irr8[t + 1], self.irr_demand_dict["irr_demand9"][moy[t] - 1],
                 9, self.irr_district_idx])  # compute the irrigation water diversion volume [m3/s] #

            qTotIN_1 = qTotIN

            # TIME-SEPARABLE OBJECTIVES

            # HYDROPOWER PRODUCTION (MWh/day)
            # Itezhitezhi
            h_itt[t] = self.Itezhitezhi.storage_to_level(s_itt[t])
            qTurb_Temp = min(r_itt[t + 1], 2 * 306)

            qturb_ReservoirSim['itt'].write(str(qTurb_Temp) + "\n")

            headTemp = (40.50 - (1030.5 - h_itt[t]))
            hydTemp = ((qTurb_Temp * headTemp * 1000 * 9.81 * 0.89 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year]
            hydTemp_dist = abs(hydTemp - self.tp_Itt[moy[t] - 1])
            gg_hydITT = np.append(gg_hydITT, hydTemp_dist)
            hydropower_def_ReservoirSim['itt'].write(str(hydTemp_dist) + "\n")

            # Kafue Gorge Upper
            h_kgu[t] = self.KafueGorgeUpper.storage_to_level(s_kgu[t])
            qTurb_Temp = min(r_kgu[t + 1], 6 * 42)

            qturb_ReservoirSim['kgu'].write(str(qTurb_Temp) + "\n")

            headTemp = (397 - (977.6 - h_kgu[t]))
            hydTemp = ((qTurb_Temp * headTemp * 1000 * 9.81 * 0.61 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year] #
            hydTemp_dist = abs(hydTemp - self.tp_Kgu[moy[t] - 1])
            gg_hydKGU = np.append(gg_hydKGU, hydTemp_dist)  #
            hydropower_def_ReservoirSim['kgu'].write(str(hydTemp_dist) + "\n")

            # Kariba North
            h_ka[t] = self.Kariba.storage_to_level(s_ka[t])  #
            qTurb_Temp_N = min(r_ka[t + 1] * 0.488,
                               6 * 200)  # Kariba North has an efficiency of 48% -. 49% of the total release goes through Kariba North #
            headTemp = (108 - (489.5 - h_ka[t]))  #
            hydTemp_N = ((qTurb_Temp_N * headTemp * 1000 * 9.81 * 0.48 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year] #

            # Kariba South
            qTurb_Temp_S = min(r_ka[t + 1] * 0.512, 6 * 140)  #

            ### Diff 5, repeated for each reservoir
            qturb_ReservoirSim['ka'].write(str(qTurb_Temp_N + qTurb_Temp_S) + "\n")
            ###

            headTemp = (110 - (489.5 - h_ka[t]))  #
            hydTemp_S = ((qTurb_Temp_S * headTemp * 1000 * 9.81 * 0.51 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year] #

            hydTemp = hydTemp_N + hydTemp_S  #
            hydTemp_dist = abs(hydTemp - self.tp_Ka[moy[t] - 1])
            gg_hydKA = np.append(gg_hydKA, hydTemp_dist)  #
            hydropower_def_ReservoirSim['ka'].write(str(hydTemp_dist) + "\n")

            # Cahora Bassa
            h_cb[t] = self.CahoraBassa.storage_to_level(s_cb[t])  #
            qTurb_Temp = min(r_cb[t + 1], 5 * 452)  #

            qturb_ReservoirSim['cb'].write(str(qTurb_Temp) + "\n")

            headTemp = (128 - (331 - h_cb[t]))  #
            hydTemp = ((qTurb_Temp * headTemp * 1000 * 9.81 * 0.73 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year] #
            hydTemp_dist = abs(hydTemp - self.tp_Cb[moy[t] - 1])
            gg_hydCB = np.append(gg_hydCB, hydTemp_dist)  #
            hydropower_def_ReservoirSim['cb'].write(str(hydTemp_dist) + "\n")

            # Kafue Gorge Lower
            h_kgl[t] = self.KafueGorgeLower.storage_to_level(s_kgl[t])  #
            qTurb_Temp = min(r_kgl[t + 1], 97.4 * 5)  #

            qturb_ReservoirSim['kgl'].write(str(qTurb_Temp) + "\n")

            headTemp = (182.7 - (586 - h_kgl[t]))  #
            hydTemp = ((qTurb_Temp * headTemp * 1000 * 9.81 * 0.88 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year] #
            hydTemp_dist = abs(hydTemp - self.tp_Kgl[moy[t] - 1])
            gg_hydKGL = np.append(gg_hydKGL, hydTemp_dist)  #
            hydropower_def_ReservoirSim['kgl'].write(str(hydTemp_dist) + "\n")

            # Victoria Falls (ROR)
            qTurb_Temp = min(max(q_Bg + q_Cuando - self.MEF_VictoriaFalls[moy[t] - 1], 0),
                             (5 * 1.2 + 6 * 12 + 6 * 12))  #
            headTemp = 100  #
            hydTemp = ((qTurb_Temp * headTemp * 1000 * 9.81 * 0.88 * (
                        24 * self.integrationStep[moy[t] - 1])) / 1000000) * 12 / 1000000  # [TWh/year] #
            gg_hydVF = np.append(gg_hydVF, hydTemp)  #

            ### Dif 6
            mass_balance_ReservoirSim['itt'].write(
                str(q_Itt) + " " + str(h_itt[t]) + " " + str(s_itt[t]) + " " + str(s_itt[t + 1]) + " " + str(
                    r_itt[t + 1]) + " " + str(r_itt_delay[t + 1]) + " " + str(gg_hydITT[t]) + "\n")
            mass_balance_ReservoirSim['kgu'].write(
                str(q_KafueFlats + r_itt_delay[t + 1] - r_irr4[t + 1]) + " " + str(h_kgu[t]) + " " + str(
                    s_kgu[t]) + " " + str(s_kgu[t + 1]) + " " + str(r_kgu[t + 1]) + " " + str(gg_hydKGU[t]) + '\n')  #
            mass_balance_ReservoirSim['kgl'].write(
                str(r_kgu[t + 1]) + " " + str(h_kgl[t]) + " " + str(s_kgl[t]) + " " + str(s_kgl[t + 1]) + " " + str(
                    r_kgl[t + 1]) + " " + str(gg_hydKGL[t]) + '\n')  #
            mass_balance_ReservoirSim['ka'].write(
                str(q_Bg + q_Cuando + q_KaLat - r_irr2[t + 1]) + " " + str(h_ka[t]) + " " + str(s_ka[t]) + " " + str(
                    s_ka[t + 1]) + " " + str(r_ka[t + 1]) + " " + str(gg_hydKA[t]) +'\n')  #
            mass_balance_ReservoirSim['cb'].write(
                str(q_Cb + r_kgl[t + 1] + r_ka[t + 1] - (r_irr3[t + 1] + r_irr5[t + 1] + r_irr6[t + 1])) + " " + str(
                    h_cb[t]) + " " + str(s_cb[t]) + " " + str(s_cb[t + 1]) + " " + str(r_cb[t + 1]) + " " + str(gg_hydCB[t]) +'\n')  #

            rDelta.write(str(r_cb[t + 1] - r_irr7[t + 1] - r_irr8[t + 1] + q_Shire - r_irr9[t + 1]) + '\n')
            irrigation.write(str(r_irr2[t + 1]) + " " + str(r_irr3[t + 1]) + " " + str(r_irr4[t + 1]) + " " + str(
                r_irr5[t + 1]) + " " + str(r_irr6[t + 1]) + " " + str(r_irr7[t + 1]) + " " + str(
                r_irr8[t + 1]) + " " + str(r_irr9[t + 1]) + '\n')
            '''
            hydropower['itt'].write(str(gg_hydITT[t] + '\n'))
            hydropower['kgu'].write(str(gg_hydKGU[t] + '\n'))
            hydropower['kgl'].write(str(gg_hydKGL[t] + '\n'))
            hydropower['ka'].write(str(gg_hydKA[t] + '\n'))
            hydropower['cb'].write(str(gg_hydCB[t] + '\n'))
            '''
            ### end dif

            # DIFF 2 cases
            deficitHYD_tot = np.append(deficitHYD_tot,
                                       gg_hydITT[t] + gg_hydKGU[t] + gg_hydKA[t] + gg_hydCB[t] + gg_hydKGL[
                                           t])  # energy production

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand2"][moy[t] - 1] - r_irr2[t + 1], 0), 2)
            gg_irr2 = np.append(gg_irr2, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand2"][moy[t] - 1])
            gg_irr2_NormDef = np.append(gg_irr2_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand3"][moy[t] - 1] - r_irr3[t + 1], 0), 2)
            gg_irr3 = np.append(gg_irr3, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand3"][moy[t] - 1])
            gg_irr3_NormDef = np.append(gg_irr3_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand4"][moy[t] - 1] - r_irr4[t + 1], 0), 2)
            gg_irr4 = np.append(gg_irr4, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand4"][moy[t] - 1])
            gg_irr4_NormDef = np.append(gg_irr4_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand5"][moy[t] - 1] - r_irr5[t + 1], 0), 2)
            gg_irr5 = np.append(gg_irr5, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand5"][moy[t] - 1])
            gg_irr5_NormDef = np.append(gg_irr5_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand6"][moy[t] - 1] - r_irr6[t + 1], 0), 2)
            gg_irr6 = np.append(gg_irr6, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand6"][moy[t] - 1])
            gg_irr6_NormDef = np.append(gg_irr6_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand7"][moy[t] - 1] - r_irr7[t + 1], 0), 2)
            gg_irr7 = np.append(gg_irr7, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand7"][moy[t] - 1])
            gg_irr7_NormDef = np.append(gg_irr7_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand8"][moy[t] - 1] - r_irr8[t + 1], 0), 2)
            gg_irr8 = np.append(gg_irr8, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand8"][moy[t] - 1])
            gg_irr8_NormDef = np.append(gg_irr8_NormDef, irrDefNorm_Temp)

            irrDef_Temp = pow(max(self.irr_demand_dict["irr_demand9"][moy[t] - 1] - r_irr9[t + 1], 0), 2)
            gg_irr9 = np.append(gg_irr9, irrDef_Temp)  # SQUARED irrigation deficit
            irrDefNorm_Temp = self.g_deficit_norm(irrDef_Temp, self.irr_demand_dict["irr_demand9"][moy[t] - 1])
            gg_irr9_NormDef = np.append(gg_irr9_NormDef, irrDefNorm_Temp)

            # DIFF 3 cases
            deficitIRR_tot = np.append(deficitIRR_tot,
                                       gg_irr2_NormDef[t] + gg_irr3_NormDef[t] + gg_irr4_NormDef[t] + gg_irr5_NormDef[
                                           t] + gg_irr6_NormDef[t] + gg_irr7_NormDef[t] + gg_irr8_NormDef[t] +
                                       gg_irr9_NormDef[t])  # SQUARED irrigation deficit

            # DELTA ENVIRONMENT DEFICIT 
            envDef_Temp = pow(
                max(self.qDelta[moy[t] - 1] - (r_cb[t + 1] - r_irr7[t + 1] - r_irr8[t + 1] + q_Shire - r_irr9[t + 1]),
                    0), 2)
            gg_env = np.append(gg_env, envDef_Temp)

            deficitENV_tot = np.append(deficitENV_tot, gg_env[t])  # Delta environment deficit

            # clear
            input = np.empty(0)
            uu = np.empty(0)

        ### Dif 7
        for reservoir in ['cb', 'itt', 'ka', 'kgu', 'kgl']:
            qturb_ReservoirSim[reservoir].close()
            mass_balance_ReservoirSim[reservoir].close()
            hydropower_def_ReservoirSim[reservoir].close()

        irrigation.close()
        rDelta.close()
        ###

        # NOT Super clear if below implementation is correct. Check!!!!
        # time-aggregation = average of step costs starting from month 1 (i.e., January 1974) // 
        # DIFF 4 cases
        JJ = np.empty(0)
        JJ = np.append(JJ, np.mean(deficitHYD_tot))
        JJ = np.append(JJ, np.mean(deficitENV_tot))
        JJ = np.append(JJ, np.mean(deficitIRR_tot))

        JJ = np.append(JJ, np.mean(gg_irr2_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr3_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr4_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr5_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr6_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr7_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr8_NormDef))
        JJ = np.append(JJ, np.mean(gg_irr9_NormDef))
        JJ = np.append(JJ, np.mean(gg_hydITT))
        JJ = np.append(JJ, np.mean(gg_hydKGU))
        JJ = np.append(JJ, np.mean(gg_hydKA))
        JJ = np.append(JJ, np.mean(gg_hydCB))
        JJ = np.append(JJ, np.mean(gg_hydKGL))
        return JJ

    # Deficit
    def g_deficit(self, q, w):

        d = w - q
        if (d < 0.0):
            d = 0.0

        return d * d

    # Normalized SQUARED deficit
    def g_deficit_norm(self, defp, w):
        """Takes two floats and divides the first by
        the square of the second.

        Parameters
        ----------
        defp : float
        w : float
            
        Returns
        -------
        def_norm : float
        """

        def_norm = 0
        if (w == 0.0):
            def_norm = 0.0
        else:
            def_norm = defp / (pow(w, 2))

        return def_norm

    def readFileSettings(self):

        def nested_getattr(object, nested_attr_list):

            obj_copy = object
            for item in nested_attr_list:
                obj_copy = getattr(obj_copy, item)
            return obj_copy

        input_model = pd.read_excel("../settings/excel_settings.xlsx", usecols=["AttributeName", "Value", "Type"],
                                    sheet_name="ModelParameters", skiprows=3)

        input_policy = pd.read_excel("../settings/excel_settings.xlsx", usecols=["AttributeName", "Value", "Type"],
                                     sheet_name="PolicyParameters", skiprows=3)

        input_df = pd.concat([input_model, input_policy], ignore_index=True)

        for _, row in input_df.iterrows():

            attribute_name_list = row["AttributeName"].split(".")
            if len(attribute_name_list) == 1:
                object, name = self, attribute_name_list[0]

            else:
                name = attribute_name_list.pop(-1)
                object = nested_getattr(self, attribute_name_list)

            if row.Type == "int":
                setattr(object, name, int(row["Value"]))
            elif row.Type == "float":
                setattr(object, name, float(row["Value"]))
            elif row.Type == "np.array":
                value = np.array(object=(row.Value.replace(" ", "")).split(";"), dtype=float)
                setattr(object, name, value)
            elif row.Type == "str":
                setattr(object, name, str(row["Value"]))

        self.integrationStep = utils.loadIntVector("../data/number_days_month.txt", self.T)
        self.integrationStep_delay = utils.loadIntVector("../data/number_days_month_delay.txt", self.T)

        # self.moy_file = utils.loadIntVector("../data/moy_1986_2005.txt", self.H)

        self.catchment_param_dict["Itt_catch_param"].CM = 1
        self.catchment_param_dict["Itt_catch_param"].inflow_file.file_name = "../data/qInfItt_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Itt_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["KafueFlats_catch_param"].CM = 1
        self.catchment_param_dict[
            "KafueFlats_catch_param"].inflow_file.file_name = "../data/qKafueFlats_1January1986_31Dec2005.txt"
        self.catchment_param_dict["KafueFlats_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Ka_catch_param"].CM = 1
        self.catchment_param_dict[
            "Ka_catch_param"].inflow_file.file_name = "../data/qInfKaLat_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Ka_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Cb_catch_param"].CM = 1
        self.catchment_param_dict["Cb_catch_param"].inflow_file.file_name = "../data/qInfCb_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Cb_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Cuando_catch_param"].CM = 1
        self.catchment_param_dict[
            "Cuando_catch_param"].inflow_file.file_name = "../data/qCuando_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Cuando_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Shire_catch_param"].CM = 1
        self.catchment_param_dict[
            "Shire_catch_param"].inflow_file.file_name = "../data/qShire_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Shire_catch_param"].inflow_file.row = self.H

        self.catchment_param_dict["Bg_catch_param"].CM = 1
        self.catchment_param_dict["Bg_catch_param"].inflow_file.file_name = "../data/qInfBg_1January1986_31Dec2005.txt"
        self.catchment_param_dict["Bg_catch_param"].inflow_file.row = self.H


# struct
class policy_parameters_construct:

    def __init__(self):
        self.tPolicy = int()
        self.policyInput = int()
        self.policyOutput = int()
        self.policyStr = int()

        self.mIn, self.mOut, self.MIn, self.MOut = tuple(4 * [np.empty(0)])
        self.muIn, self.muOut, self.stdIn, self.stdOut = tuple(4 * [np.empty(0)])


class irr_function_parameters:
    def __init__(self):
        self.num_irr = int()
        self.mParam = np.empty(0)
        self.MParam = np.empty(0)
