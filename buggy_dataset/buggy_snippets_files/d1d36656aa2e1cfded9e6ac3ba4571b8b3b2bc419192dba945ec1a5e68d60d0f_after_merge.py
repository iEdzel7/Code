def dump(dbs, f, **options):
    if 'arVersion' in options:
        arVersion = options["arVersion"]
    else:
        arVersion = "3.2.3"

    for name in dbs:
        db = dbs[name]
        for frame in db.frames:
            for signal in frame.signals:
                for rec in signal.receiver:
                    if rec.strip() not in frame.receiver:
                        frame.receiver.append(rec.strip())

    if arVersion[0] == "3":
        xsi = 'http://www.w3.org/2001/XMLSchema-instance'
        root = etree.Element(
            'AUTOSAR',
            nsmap={
                None: 'http://autosar.org/' + arVersion,
                'xsi': xsi})
        root.attrib['{{{pre}}}schemaLocation'.format(
            pre=xsi)] = 'http://autosar.org/' + arVersion + ' AUTOSAR_' + arVersion.replace('.', '') + '.xsd'
        toplevelPackages = createSubElement(root, 'TOP-LEVEL-PACKAGES')
    else:
        xsi = 'http://www.w3.org/2001/XMLSchema-instance'
        root = etree.Element(
            'AUTOSAR',
            nsmap={
                None: "http://autosar.org/schema/r4.0",
                'xsi': xsi})
        root.attrib['{{{pre}}}schemaLocation'.format(
            pre=xsi)] = 'http://autosar.org/schema/r4.0 AUTOSAR_' + arVersion.replace('.', '-') + '.xsd'
        toplevelPackages = createSubElement(root, 'AR-PACKAGES')

    #
    # AR-PACKAGE Cluster
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'Cluster')
    elements = createSubElement(arPackage, 'ELEMENTS')

    for name in dbs:
        db = dbs[name]
#        if len(name) == 0:
#            (path, ext) = os.path.splitext(filename)
#            busName = path
#        else:
        if len(name) > 0:
            busName = name
        else:
            busName = "CAN"

        cancluster = createSubElement(elements, 'CAN-CLUSTER')
        createSubElement(cancluster, 'SHORT-NAME', busName)
        if arVersion[0] == "3":
         #       createSubElement(cancluster, 'SPEED', '50000')
            physicalChannels = createSubElement(
                cancluster, 'PHYSICAL-CHANNELS')
            physicalChannel = createSubElement(
                physicalChannels, 'PHYSICAL-CHANNEL')
            createSubElement(physicalChannel, 'SHORT-NAME', 'CAN')
            frameTriggering = createSubElement(
                physicalChannel, 'FRAME-TRIGGERINGSS')
        else:
            canClusterVaraints = createSubElement(
                cancluster, 'CAN-CLUSTER-VARIANTS')
            canClusterConditional = createSubElement(
                canClusterVaraints, 'CAN-CLUSTER-CONDITIONAL')
            physicalChannels = createSubElement(
                canClusterConditional, 'PHYSICAL-CHANNELS')
            physicalChannel = createSubElement(
                physicalChannels, 'CAN-PHYSICAL-CHANNEL')
            createSubElement(physicalChannel, 'SHORT-NAME', 'CAN')
            frameTriggering = createSubElement(
                physicalChannel, 'FRAME-TRIGGERINGS')
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                logger.error("export complex multiplexers is not supported - ignoring frame " + frame.name)
                continue
            canFrameTriggering = createSubElement(
                frameTriggering, 'CAN-FRAME-TRIGGERING')
            createSubElement(canFrameTriggering, 'SHORT-NAME', frame.name)
            framePortRefs = createSubElement(
                canFrameTriggering, 'FRAME-PORT-REFS')
            for transmitter in frame.transmitter:
                framePortRef = createSubElement(
                    framePortRefs, 'FRAME-PORT-REF')
                framePortRef.set('DEST', 'FRAME-PORT')
                framePortRef.text = "/ECU/" + transmitter + \
                    "/CN_" + transmitter + "/" + frame.name
            for rec in frame.receiver:
                framePortRef = createSubElement(
                    framePortRefs, 'FRAME-PORT-REF')
                framePortRef.set('DEST', 'FRAME-PORT')
                framePortRef.text = "/ECU/" + rec + "/CN_" + rec + "/" + frame.name
            frameRef = createSubElement(canFrameTriggering, 'FRAME-REF')
            if arVersion[0] == "3":
                frameRef.set('DEST', 'FRAME')
                frameRef.text = "/Frame/FRAME_" + frame.name
                pduTriggeringRefs = createSubElement(
                    canFrameTriggering, 'I-PDU-TRIGGERING-REFS')
                pduTriggeringRef = createSubElement(
                    pduTriggeringRefs, 'I-PDU-TRIGGERING-REF')
                pduTriggeringRef.set('DEST', 'I-PDU-TRIGGERING')
            else:
                frameRef.set('DEST', 'CAN-FRAME')
                frameRef.text = "/CanFrame/FRAME_" + frame.name
                pduTriggering = createSubElement(
                    canFrameTriggering, 'PDU-TRIGGERINGS')
                pduTriggeringRefConditional = createSubElement(
                    pduTriggering, 'PDU-TRIGGERING-REF-CONDITIONAL')
                pduTriggeringRef = createSubElement(
                    pduTriggeringRefConditional, 'PDU-TRIGGERING-REF')
                pduTriggeringRef.set('DEST', 'PDU-TRIGGERING')

            if frame.extended == 0:
                createSubElement(
                    canFrameTriggering,
                    'CAN-ADDRESSING-MODE',
                    'STANDARD')
            else:
                createSubElement(
                    canFrameTriggering,
                    'CAN-ADDRESSING-MODE',
                    'EXTENDED')
            createSubElement(canFrameTriggering, 'IDENTIFIER', str(frame.id))

            pduTriggeringRef.text = "/Cluster/CAN/IPDUTRIGG_" + frame.name

        if arVersion[0] == "3":
            ipduTriggerings = createSubElement(
                physicalChannel, 'I-PDU-TRIGGERINGS')
            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                ipduTriggering = createSubElement(
                    ipduTriggerings, 'I-PDU-TRIGGERING')
                createSubElement(
                    ipduTriggering,
                    'SHORT-NAME',
                    "IPDUTRIGG_" +
                    frame.name)
                ipduRef = createSubElement(ipduTriggering, 'I-PDU-REF')
                ipduRef.set('DEST', 'SIGNAL-I-PDU')
                ipduRef.text = "/PDU/PDU_" + frame.name
            isignalTriggerings = createSubElement(
                physicalChannel, 'I-SIGNAL-TRIGGERINGS')
            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                for signal in frame.signals:
                    isignalTriggering = createSubElement(
                        isignalTriggerings, 'I-SIGNAL-TRIGGERING')
                    createSubElement(isignalTriggering,
                                     'SHORT-NAME', signal.name)
                    iSignalPortRefs = createSubElement(
                        isignalTriggering, 'I-SIGNAL-PORT-REFS')

                    for receiver in signal.receiver:
                        iSignalPortRef = createSubElement(
                            iSignalPortRefs,
                            'I-SIGNAL-PORT-REF',
                            '/ECU/' +
                            receiver +
                            '/CN_' +
                            receiver +
                            '/' +
                            signal.name)
                        iSignalPortRef.set('DEST', 'SIGNAL-PORT')

                    isignalRef = createSubElement(
                        isignalTriggering, 'SIGNAL-REF')
                    isignalRef.set('DEST', 'I-SIGNAL')
                    isignalRef.text = "/ISignal/" + signal.name
        else:
            isignalTriggerings = createSubElement(
                physicalChannel, 'I-SIGNAL-TRIGGERINGS')
            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                for signal in frame.signals:
                    isignalTriggering = createSubElement(
                        isignalTriggerings, 'I-SIGNAL-TRIGGERING')
                    createSubElement(isignalTriggering,
                                     'SHORT-NAME', signal.name)
                    iSignalPortRefs = createSubElement(
                        isignalTriggering, 'I-SIGNAL-PORT-REFS')
                    for receiver in signal.receiver:
                        iSignalPortRef = createSubElement(
                            iSignalPortRefs,
                            'I-SIGNAL-PORT-REF',
                            '/ECU/' +
                            receiver +
                            '/CN_' +
                            receiver +
                            '/' +
                            signal.name)
                        iSignalPortRef.set('DEST', 'I-SIGNAL-PORT')

                    isignalRef = createSubElement(
                        isignalTriggering, 'I-SIGNAL-REF')
                    isignalRef.set('DEST', 'I-SIGNAL')
                    isignalRef.text = "/ISignal/" + signal.name
            ipduTriggerings = createSubElement(
                physicalChannel, 'PDU-TRIGGERINGS')
            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                ipduTriggering = createSubElement(
                    ipduTriggerings, 'PDU-TRIGGERING')
                createSubElement(
                    ipduTriggering,
                    'SHORT-NAME',
                    "IPDUTRIGG_" +
                    frame.name)
                # missing: I-PDU-PORT-REFS
                ipduRef = createSubElement(ipduTriggering, 'I-PDU-REF')
                ipduRef.set('DEST', 'I-SIGNAL-I-PDU')
                ipduRef.text = "/PDU/PDU_" + frame.name
                # missing: I-SIGNAL-TRIGGERINGS

# TODO
#        ipduTriggerings = createSubElement(physicalChannel, 'PDU-TRIGGERINGS')
#        for frame in db.frames:
#            ipduTriggering = createSubElement(ipduTriggerings, 'PDU-TRIGGERING')
#            createSubElement(ipduTriggering, 'SHORT-NAME', "PDUTRIGG_" + frame.name)
#            ipduRef = createSubElement(ipduTriggering, 'I-PDU-REF')
#            ipduRef.set('DEST','SIGNAL-I-PDU')
#            ipduRef.text = "/PDU/PDU_" + frame.name

    #
    # AR-PACKAGE FRAME
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    if arVersion[0] == "3":
        createSubElement(arPackage, 'SHORT-NAME', 'Frame')
    else:
        createSubElement(arPackage, 'SHORT-NAME', 'CanFrame')

    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        # TODO: reused frames will be paced multiple times in file
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                continue

            if arVersion[0] == "3":
                frameEle = createSubElement(elements, 'FRAME')
            else:
                frameEle = createSubElement(elements, 'CAN-FRAME')
            createSubElement(frameEle, 'SHORT-NAME', "FRAME_" + frame.name)
            if frame.comment:
                desc = createSubElement(frameEle, 'DESC')
                l2 = createSubElement(desc, 'L-2')
                l2.set("L", "FOR-ALL")
                l2.text = frame.comment
            createSubElement(frameEle, 'FRAME-LENGTH', "%d" % frame.size)
            pdumappings = createSubElement(frameEle, 'PDU-TO-FRAME-MAPPINGS')
            pdumapping = createSubElement(pdumappings, 'PDU-TO-FRAME-MAPPING')
            createSubElement(pdumapping, 'SHORT-NAME', frame.name)
            createSubElement(
                pdumapping,
                'PACKING-BYTE-ORDER',
                "MOST-SIGNIFICANT-BYTE-LAST")
            pduRef = createSubElement(pdumapping, 'PDU-REF')
            createSubElement(pdumapping, 'START-POSITION', '0')
            pduRef.text = "/PDU/PDU_" + frame.name
            if arVersion[0] == "3":
                pduRef.set('DEST', 'SIGNAL-I-PDU')
            else:
                pduRef.set('DEST', 'I-SIGNAL-I-PDU')

    #
    # AR-PACKAGE PDU
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'PDU')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                continue

            if arVersion[0] == "3":
                signalIpdu = createSubElement(elements, 'SIGNAL-I-PDU')
                createSubElement(signalIpdu, 'SHORT-NAME', "PDU_" + frame.name)
                createSubElement(signalIpdu, 'LENGTH', "%d" %
                                 int(frame.size * 8))
            else:
                signalIpdu = createSubElement(elements, 'I-SIGNAL-I-PDU')
                createSubElement(signalIpdu, 'SHORT-NAME', "PDU_" + frame.name)
                createSubElement(signalIpdu, 'LENGTH', "%d" % int(frame.size))

            # I-PDU-TIMING-SPECIFICATION
            if arVersion[0] == "3":
                signalToPduMappings = createSubElement(
                    signalIpdu, 'SIGNAL-TO-PDU-MAPPINGS')
            else:
                signalToPduMappings = createSubElement(
                    signalIpdu, 'I-SIGNAL-TO-PDU-MAPPINGS')

            for signal in frame.signals:
                signalToPduMapping = createSubElement(
                    signalToPduMappings, 'I-SIGNAL-TO-I-PDU-MAPPING')
                createSubElement(signalToPduMapping, 'SHORT-NAME', signal.name)

                if arVersion[0] == "3":
                    if signal.is_little_endian == 1:  # Intel
                        createSubElement(
                            signalToPduMapping,
                            'PACKING-BYTE-ORDER',
                            'MOST-SIGNIFICANT-BYTE-LAST')
                    else:  # Motorola
                        createSubElement(
                            signalToPduMapping,
                            'PACKING-BYTE-ORDER',
                            'MOST-SIGNIFICANT-BYTE-FIRST')
                    signalRef = createSubElement(
                        signalToPduMapping, 'SIGNAL-REF')
                else:
                    signalRef = createSubElement(
                        signalToPduMapping, 'I-SIGNAL-REF')
                    if signal.is_little_endian == 1:  # Intel
                        createSubElement(
                            signalToPduMapping,
                            'PACKING-BYTE-ORDER',
                            'MOST-SIGNIFICANT-BYTE-LAST')
                    else:  # Motorola
                        createSubElement(
                            signalToPduMapping,
                            'PACKING-BYTE-ORDER',
                            'MOST-SIGNIFICANT-BYTE-FIRST')
                signalRef.text = "/ISignal/" + signal.name
                signalRef.set('DEST', 'I-SIGNAL')

                createSubElement(signalToPduMapping, 'START-POSITION',
                                 str(signal.getStartbit(bitNumbering=1)))
                # missing: TRANSFER-PROPERTY: PENDING/...

            for group in frame.signalGroups:
                signalToPduMapping = createSubElement(
                    signalToPduMappings, 'I-SIGNAL-TO-I-PDU-MAPPING')
                createSubElement(signalToPduMapping, 'SHORT-NAME', group.name)
                signalRef = createSubElement(signalToPduMapping, 'SIGNAL-REF')
                signalRef.text = "/ISignal/" + group.name
                signalRef.set('DEST', 'I-SIGNAL')
                # TODO: TRANSFER-PROPERTY: PENDING???

    #
    # AR-PACKAGE ISignal
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'ISignal')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                continue

            for signal in frame.signals:
                signalEle = createSubElement(elements, 'I-SIGNAL')
                createSubElement(signalEle, 'SHORT-NAME', signal.name)
                if arVersion[0] == "4":
                    createSubElement(signalEle, 'LENGTH',
                                     str(signal.signalsize))

                    networkRepresentProps = createSubElement(
                        signalEle, 'NETWORK-REPRESENTATION-PROPS')
                    swDataDefPropsVariants = createSubElement(
                        networkRepresentProps, 'SW-DATA-DEF-PROPS-VARIANTS')
                    swDataDefPropsConditional = createSubElement(
                        swDataDefPropsVariants, 'SW-DATA-DEF-PROPS-CONDITIONAL')
                    
                    baseTypeRef = createSubElement(swDataDefPropsConditional, 'BASE-TYPE-REF')
                    baseTypeRef.set('DEST', 'SW-BASE-TYPE')
                    createType, size = getBaseTypeOfSignal(signal)
                    baseTypeRef.text = "/DataType/" + createType
                    compuMethodRef = createSubElement(
                        swDataDefPropsConditional,
                        'COMPU-METHOD-REF',
                        '/DataType/Semantics/' + signal.name)
                    compuMethodRef.set('DEST', 'COMPU-METHOD')
                    unitRef = createSubElement(
                        swDataDefPropsConditional,
                        'UNIT-REF',
                        '/DataType/Unit/' + signal.name)
                    unitRef.set('DEST', 'UNIT')

                sysSigRef = createSubElement(signalEle, 'SYSTEM-SIGNAL-REF')
                sysSigRef.text = "/Signal/" + signal.name

                sysSigRef.set('DEST', 'SYSTEM-SIGNAL')
            for group in frame.signalGroups:
                signalEle = createSubElement(elements, 'I-SIGNAL')
                createSubElement(signalEle, 'SHORT-NAME', group.name)
                sysSigRef = createSubElement(signalEle, 'SYSTEM-SIGNAL-REF')
                sysSigRef.text = "/Signal/" + group.name
                sysSigRef.set('DEST', 'SYSTEM-SIGNAL-GROUP')

    #
    # AR-PACKAGE Signal
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'Signal')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                continue

            for signal in frame.signals:
                signalEle = createSubElement(elements, 'SYSTEM-SIGNAL')
                createSubElement(signalEle, 'SHORT-NAME', signal.name)
                if signal.comment:
                    desc = createSubElement(signalEle, 'DESC')
                    l2 = createSubElement(desc, 'L-2')
                    l2.set("L", "FOR-ALL")
                    l2.text = signal.comment
                if arVersion[0] == "3":
                    dataTypeRef = createSubElement(signalEle, 'DATA-TYPE-REF')
                    if signal.is_float:
                        dataTypeRef.set('DEST', 'REAL-TYPE')
                    else:
                        dataTypeRef.set('DEST', 'INTEGER-TYPE')
                    dataTypeRef.text = "/DataType/" + signal.name
                    createSubElement(signalEle, 'LENGTH',
                                     str(signal.signalsize))
            for group in frame.signalGroups:
                groupEle = createSubElement(elements, 'SYSTEM-SIGNAL-GROUP')
                createSubElement(signalEle, 'SHORT-NAME', group.name)
                if arVersion[0] == "3":
                    dataTypeRef.set('DEST', 'INTEGER-TYPE')
                sysSignalRefs = createSubElement(
                    groupEle, 'SYSTEM-SIGNAL-REFS')
                for member in group.signals:
                    memberEle = createSubElement(
                        sysSignalRefs, 'SYSTEM-SIGNAL-REF')
                    memberEle.set('DEST', 'SYSTEM-SIGNAL')
                    memberEle.text = "/Signal/" + member.name

#                       initValueRef = createSubElement(signalEle, 'INIT-VALUE-REF')
#                       initValueRef.set('DEST','INTEGER-LITERAL')
#                       initValueRef.text = "/CONSTANTS/" + signal.name

    #
    # AR-PACKAGE Datatype
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'DataType')
    elements = createSubElement(arPackage, 'ELEMENTS')

    if arVersion[0] == "3":
        for name in dbs:
            db = dbs[name]
            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                for signal in frame.signals:
                    if signal.is_float:
                        typeEle = createSubElement(elements, 'REAL-TYPE')
                    else:
                        typeEle = createSubElement(elements, 'INTEGER-TYPE')
                    createSubElement(typeEle, 'SHORT-NAME', signal.name)
                    swDataDefProps = createSubElement(
                        typeEle, 'SW-DATA-DEF-PROPS')
                    if signal.is_float:
                        encoding = createSubElement(typeEle, 'ENCODING')                        
                        if signal.signalsize > 32:
                            encoding.text = "DOUBLE"
                        else:
                            encoding.text = "SINGLE"
                    compuMethodRef = createSubElement(
                        swDataDefProps, 'COMPU-METHOD-REF')
                    compuMethodRef.set('DEST', 'COMPU-METHOD')
                    compuMethodRef.text = "/DataType/Semantics/" + signal.name
    else:
        createdTypes = []
        for name in dbs:
            db = dbs[name]
            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                for signal in frame.signals:
                    createType, size = getBaseTypeOfSignal(signal)
                    if createType not in createdTypes:
                        createdTypes.append(createType)
                        swBaseType = createSubElement(elements, 'SW-BASE-TYPE')
                        sname = createSubElement(swBaseType, 'SHORT-NAME')
                        sname.text = createType
                        cat = createSubElement(swBaseType, 'CATEGORY')
                        cat.text = "FIXED_LENGTH"
                        baseTypeSize = createSubElement(swBaseType, 'BASE-TYPE-SIZE')
                        baseTypeSize.text = str(size)
                        if signal.is_float:
                            enc = createSubElement(swBaseType, 'BASE-TYPE-ENCODING')
                            enc.text = "IEEE754"

    if arVersion[0] == "3":
        subpackages = createSubElement(arPackage, 'SUB-PACKAGES')
    else:
        subpackages = createSubElement(arPackage, 'AR-PACKAGES')
    arPackage = createSubElement(subpackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'Semantics')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                continue

            for signal in frame.signals:
                compuMethod = createSubElement(elements, 'COMPU-METHOD')
                createSubElement(compuMethod, 'SHORT-NAME', signal.name)
                # missing: UNIT-REF
                compuIntToPhys = createSubElement(
                    compuMethod, 'COMPU-INTERNAL-TO-PHYS')
                compuScales = createSubElement(compuIntToPhys, 'COMPU-SCALES')
                for value in sorted(signal.values, key=lambda x: int(x)):
                    compuScale = createSubElement(compuScales, 'COMPU-SCALE')
                    desc = createSubElement(compuScale, 'DESC')
                    l2 = createSubElement(desc, 'L-2')
                    l2.set('L', 'FOR-ALL')
                    l2.text = signal.values[value]
                    createSubElement(compuScale, 'LOWER-LIMIT', str(value))
                    createSubElement(compuScale, 'UPPER-LIMIT', str(value))
                    compuConst = createSubElement(compuScale, 'COMPU-CONST')
                    createSubElement(compuConst, 'VT', signal.values[value])
                else:
                    compuScale = createSubElement(compuScales, 'COMPU-SCALE')
    #                createSubElement(compuScale, 'LOWER-LIMIT', str(#TODO))
    #                createSubElement(compuScale, 'UPPER-LIMIT', str(#TODO))
                    compuRationslCoeff = createSubElement(
                        compuScale, 'COMPU-RATIONAL-COEFFS')
                    compuNumerator = createSubElement(
                        compuRationslCoeff, 'COMPU-NUMERATOR')
                    createSubElement(compuNumerator, 'V', "%g" % signal.offset)
                    createSubElement(compuNumerator, 'V', "%g" % signal.factor)
                    compuDenomiator = createSubElement(
                        compuRationslCoeff, 'COMPU-DENOMINATOR')
                    createSubElement(compuDenomiator, 'V', "1")

    arPackage = createSubElement(subpackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'Unit')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        for frame in db.frames:
            if frame.is_complex_multiplexed:
                continue

            for signal in frame.signals:
                unit = createSubElement(elements, 'UNIT')
                createSubElement(unit, 'SHORT-NAME', signal.name)
                createSubElement(unit, 'DISPLAY-NAME', signal.unit)

    txIPduGroups = {}
    rxIPduGroups = {}

    #
    # AR-PACKAGE ECU
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'ECU')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for name in dbs:
        db = dbs[name]
        for ecu in db.boardUnits:
            ecuInstance = createSubElement(elements, 'ECU-INSTANCE')
            createSubElement(ecuInstance, 'SHORT-NAME', ecu.name)
            if ecu.comment:
                desc = createSubElement(ecuInstance, 'DESC')
                l2 = createSubElement(desc, 'L-2')
                l2.set('L', 'FOR-ALL')
                l2.text = ecu.comment

            if arVersion[0] == "3":
                assoIpduGroupRefs = createSubElement(
                    ecuInstance, 'ASSOCIATED-I-PDU-GROUP-REFS')
                connectors = createSubElement(ecuInstance, 'CONNECTORS')
                commConnector = createSubElement(
                    connectors, 'COMMUNICATION-CONNECTOR')
            else:
                assoIpduGroupRefs = createSubElement(
                    ecuInstance, 'ASSOCIATED-COM-I-PDU-GROUP-REFS')
                connectors = createSubElement(ecuInstance, 'CONNECTORS')
                commConnector = createSubElement(
                    connectors, 'CAN-COMMUNICATION-CONNECTOR')

            createSubElement(commConnector, 'SHORT-NAME', 'CN_' + ecu.name)
            ecuCommPortInstances = createSubElement(
                commConnector, 'ECU-COMM-PORT-INSTANCES')

            recTemp = None
            sendTemp = None

            for frame in db.frames:
                if frame.is_complex_multiplexed:
                    continue

                if ecu.name in frame.transmitter:
                    frameport = createSubElement(
                        ecuCommPortInstances, 'FRAME-PORT')
                    createSubElement(frameport, 'SHORT-NAME', frame.name)
                    createSubElement(
                        frameport, 'COMMUNICATION-DIRECTION', 'OUT')
                    sendTemp = 1
                    if ecu.name + "_Tx" not in txIPduGroups:
                        txIPduGroups[ecu.name + "_Tx"] = []
                    txIPduGroups[ecu.name + "_Tx"].append(frame.name)

                    # missing I-PDU-PORT
                    for signal in frame.signals:
                        if arVersion[0] == "3":
                            signalPort = createSubElement(
                                ecuCommPortInstances, 'SIGNAL-PORT')
                        else:
                            signalPort = createSubElement(
                                ecuCommPortInstances, 'I-SIGNAL-PORT')

                        createSubElement(signalPort, 'SHORT-NAME', signal.name)
                        createSubElement(
                            signalPort, 'COMMUNICATION-DIRECTION', 'OUT')
                if ecu.name in frame.receiver:
                    frameport = createSubElement(
                        ecuCommPortInstances, 'FRAME-PORT')
                    createSubElement(frameport, 'SHORT-NAME', frame.name)
                    createSubElement(
                        frameport, 'COMMUNICATION-DIRECTION', 'IN')
                    recTemp = 1
                    if ecu.name + "_Tx" not in rxIPduGroups:
                        rxIPduGroups[ecu.name + "_Rx"] = []
                    rxIPduGroups[ecu.name + "_Rx"].append(frame.name)

                    # missing I-PDU-PORT
                    for signal in frame.signals:
                        if ecu.name in signal.receiver:
                            if arVersion[0] == "3":
                                signalPort = createSubElement(
                                    ecuCommPortInstances, 'SIGNAL-PORT')
                            else:
                                signalPort = createSubElement(
                                    ecuCommPortInstances, 'I-SIGNAL-PORT')

                            createSubElement(
                                signalPort, 'SHORT-NAME', signal.name)
                            createSubElement(
                                signalPort, 'COMMUNICATION-DIRECTION', 'IN')

            if recTemp is not None:
                if arVersion[0] == "3":
                    assoIpduGroupRef = createSubElement(
                        assoIpduGroupRefs, 'ASSOCIATED-I-PDU-GROUP-REF')
                    assoIpduGroupRef.set('DEST', "I-PDU-GROUP")
                else:
                    assoIpduGroupRef = createSubElement(
                        assoIpduGroupRefs, 'ASSOCIATED-COM-I-PDU-GROUP-REF')
                    assoIpduGroupRef.set('DEST', "I-SIGNAL-I-PDU-GROUP")

                assoIpduGroupRef.text = "/IPDUGroup/" + ecu.name + "_Rx"

            if sendTemp is not None:
                if arVersion[0] == "3":
                    assoIpduGroupRef = createSubElement(
                        assoIpduGroupRefs, 'ASSOCIATED-I-PDU-GROUP-REF')
                    assoIpduGroupRef.set('DEST', "I-PDU-GROUP")
                else:
                    assoIpduGroupRef = createSubElement(
                        assoIpduGroupRefs, 'ASSOCIATED-COM-I-PDU-GROUP-REF')
                    assoIpduGroupRef.set('DEST', "I-SIGNAL-I-PDU-GROUP")
                assoIpduGroupRef.text = "/IPDUGroup/" + ecu.name + "_Tx"

    #
    # AR-PACKAGE IPDUGroup
    #
    arPackage = createSubElement(toplevelPackages, 'AR-PACKAGE')
    createSubElement(arPackage, 'SHORT-NAME', 'IPDUGroup')
    elements = createSubElement(arPackage, 'ELEMENTS')
    for pdugrp in txIPduGroups:
        if arVersion[0] == "3":
            ipduGrp = createSubElement(elements, 'I-PDU-GROUP')
        else:
            ipduGrp = createSubElement(elements, 'I-SIGNAL-I-PDU-GROUP')

        createSubElement(ipduGrp, 'SHORT-NAME', pdugrp)
        createSubElement(ipduGrp, 'COMMUNICATION-DIRECTION', "OUT")

        if arVersion[0] == "3":
            ipduRefs = createSubElement(ipduGrp, 'I-PDU-REFS')
            for frame in txIPduGroups[pdugrp]:
                ipduRef = createSubElement(ipduRefs, 'I-PDU-REF')
                ipduRef.set('DEST', "SIGNAL-I-PDU")
                ipduRef.text = "/PDU/PDU_" + frame
        else:
            isignalipdus = createSubElement(ipduGrp, 'I-SIGNAL-I-PDUS')
            for frame in txIPduGroups[pdugrp]:
                isignalipdurefconditional = createSubElement(
                    isignalipdus, 'I-SIGNAL-I-PDU-REF-CONDITIONAL')
                ipduRef = createSubElement(
                    isignalipdurefconditional, 'I-SIGNAL-I-PDU-REF')
                ipduRef.set('DEST', "I-SIGNAL-I-PDU")
                ipduRef.text = "/PDU/PDU_" + frame

    if arVersion[0] == "3":
        for pdugrp in rxIPduGroups:
            ipduGrp = createSubElement(elements, 'I-PDU-GROUP')
            createSubElement(ipduGrp, 'SHORT-NAME', pdugrp)
            createSubElement(ipduGrp, 'COMMUNICATION-DIRECTION', "IN")

            ipduRefs = createSubElement(ipduGrp, 'I-PDU-REFS')
            for frame in rxIPduGroups[pdugrp]:
                ipduRef = createSubElement(ipduRefs, 'I-PDU-REF')
                ipduRef.set('DEST', "SIGNAL-I-PDU")
                ipduRef.text = "/PDU/PDU_" + frame
    else:
        for pdugrp in rxIPduGroups:
            ipduGrp = createSubElement(elements, 'I-SIGNAL-I-PDU-GROUP')
            createSubElement(ipduGrp, 'SHORT-NAME', pdugrp)
            createSubElement(ipduGrp, 'COMMUNICATION-DIRECTION', "IN")
            isignalipdus = createSubElement(ipduGrp, 'I-SIGNAL-I-PDUS')
            for frame in rxIPduGroups[pdugrp]:
                isignalipdurefconditional = createSubElement(
                    isignalipdus, 'I-SIGNAL-I-PDU-REF-CONDITIONAL')
                ipduRef = createSubElement(
                    isignalipdurefconditional, 'I-SIGNAL-I-PDU-REF')
                ipduRef.set('DEST', "I-SIGNAL-I-PDU")
                ipduRef.text = "/PDU/PDU_" + frame

    f.write(etree.tostring(root, pretty_print=True, xml_declaration=True))