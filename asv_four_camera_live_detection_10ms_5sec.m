clc;
clear;
close all;

%% ASV CATAMARAN - FOUR CAMERAS, SHORELINE, AND WORKING VIEW SWITCH
% MATLAB R2025b / Simulink 3D Animation sim3d
% The aerial shoreline is a visualization; the 3D shoreline is geometry.
% Live Python boundary detection exchanges frames through a shared folder.
% Start live_water_boundary_detector.py before running this MATLAB script.

disp('==============================================');
disp(' ASV FOUR-CAMERA VISION SYSTEM + SHORELINE');
disp('==============================================');

%% 1. PARAMETERS
sampleTime = 1/60;
stopTime = 10;
boatSpeed = 7.0;             % metres/second
initialBoatX = 0;
cameraHeight = 2.2;
cameraFocalLength = [180 180];
hullLength = 6;
hullWidth = 0.8;
hullHeight = 0.7;
hullSpacing = 1.1;
shoreY = 5;                   % water/land boundary in world Y coordinates

%% 2. FOUR-CAMERA DISPLAY
cameraFig = figure('Name','ASV Four-Camera Vision System', ...
    'NumberTitle','off','Color','k','Position',[50 50 1500 900]);
cameraTitles = {'FRONT CAMERA';'RIGHT CAMERA';'REAR CAMERA';'LEFT CAMERA'};
cameraAxes = gobjects(1,4);
cameraImages = gobjects(1,4);
for k = 1:4
    cameraAxes(k) = subplot(2,2,k,'Parent',cameraFig);
    cameraImages(k) = image(cameraAxes(k),zeros(720,1280,3,'uint8'));
    axis(cameraAxes(k),'image');
    axis(cameraAxes(k),'off');
    title(cameraAxes(k),cameraTitles{k},'Color','w','FontSize',14,'FontWeight','bold');
end
drawnow;

%% 3. AERIAL VIEW: BLUE WATER + GREEN SHORE + DISTINCT BOUNDARY
aerialFig = figure('Name','ASV Aerial View', ...
    'NumberTitle','off','Color','w','Position',[200 100 1100 700]);
aerialAx = axes('Parent',aerialFig,'Position',[0.10 0.15 0.86 0.78]);
hold(aerialAx,'on'); grid(aerialAx,'on'); axis(aerialAx,'equal');
xlabel(aerialAx,'X position (m)'); ylabel(aerialAx,'Y position (m)');
% Water region (blue) and land region (green), shown as top-down patches.
patch(aerialAx,[-80 120 120 -80],[-50 -50 shoreY shoreY], ...
    [0.20 0.62 0.86],'EdgeColor','none','FaceAlpha',1);
patch(aerialAx,[-80 120 120 -80],[shoreY shoreY 50 50], ...
    [0.45 0.72 0.35],'EdgeColor','none','FaceAlpha',1);
plot(aerialAx,[-80 120],[shoreY shoreY],'k-','LineWidth',2.5);
text(aerialAx,18,-8,'WATER','Color','w','FontWeight','bold','FontSize',12);
text(aerialAx,18,10,'SHORE / LAND','Color','k','FontWeight','bold','FontSize',12);

obstaclePositions = [30 0; 15 6; 15 -6; -8 0];
for k = 1:size(obstaclePositions,1)
    rectangle(aerialAx,'Position',[obstaclePositions(k,1)-0.75, ...
        obstaclePositions(k,2)-0.75,1.5,1.5], ...
        'FaceColor',[0.85 0.2 0.2],'EdgeColor','k','LineWidth',1.2);
end
hullLeft = rectangle(aerialAx,'Position',[initialBoatX-hullLength/2, ...
    -hullSpacing-hullWidth/2,hullLength,hullWidth], ...
    'FaceColor',[0.1 0.35 0.75],'EdgeColor','k');
hullRight = rectangle(aerialAx,'Position',[initialBoatX-hullLength/2, ...
    hullSpacing-hullWidth/2,hullLength,hullWidth], ...
    'FaceColor',[0.1 0.35 0.75],'EdgeColor','k');
deckView = rectangle(aerialAx,'Position',[initialBoatX-2.25,-1.6,4.5,3.2], ...
    'FaceColor',[0.70 0.70 0.70],'EdgeColor','k','LineWidth',1.2);
headingArrow = quiver(aerialAx,initialBoatX,0,3,0,0, ...
    'LineWidth',2.5,'MaxHeadSize',1,'Color',[0 0.2 0]);
xlim(aerialAx,[-12 40]); ylim(aerialAx,[-15 15]);
title(aerialAx,'ASV AERIAL VIEW — FIXED OVERVIEW','FontWeight','bold');

% A true switch: fixed overview versus a moving window centered on the boat.
setappdata(aerialFig,'AerialMode',true);
uicontrol(aerialFig,'Style','pushbutton','String','Switch Aerial / Follow View', ...
    'Position',[20 18 240 36],'FontSize',11,'Callback',@switchAerialView);

%% 4. PREPARE LIVE DETECTION EXCHANGE FOLDER
projectFolder = fileparts(fileparts(mfilename('fullpath')));
ioFolder = fullfile(projectFolder,'live_detection_io');
if ~exist(ioFolder,'dir'), mkdir(ioFolder); end
camNames = {'front','right','rear','left'};
for q = 1:4
    inputPath = fullfile(ioFolder,[camNames{q} '_input.png']);
    outputPath = fullfile(ioFolder,[camNames{q} '_overlay.png']);
    if isfile(inputPath), delete(inputPath); end
    if isfile(outputPath), delete(outputPath); end
end
setappdata(cameraFig,'LastOverlayTime',zeros(1,4));
fprintf('Live detection exchange folder: %s\n',ioFolder);
fprintf('Start the Python detector in a separate Command Prompt before/during simulation.\n');

%% 5. CREATE 3D WORLD WITH CALLBACK
world = sim3d.World(Output=@outputImpl, ...
    Update=@(w) updateSimulation(w,cameraFig,cameraImages, ...
    aerialFig,aerialAx,hullLeft,hullRight,deckView,headingArrow, ...
    boatSpeed,sampleTime,stopTime,initialBoatX,shoreY,ioFolder));

%% 6. CATAMARAN HULLS, DECK, AND PLATFORM
leftHull = sim3d.Actor(ActorName="LeftHull");
leftHull.Mobility = sim3d.utils.MobilityTypes.Movable;
leftHull.createShape("box",[hullLength hullWidth hullHeight]);
leftHull.Translation = [initialBoatX,-hullSpacing,0.6]; world.add(leftHull);

rightHull = sim3d.Actor(ActorName="RightHull");
rightHull.Mobility = sim3d.utils.MobilityTypes.Movable;
rightHull.createShape("box",[hullLength hullWidth hullHeight]);
rightHull.Translation = [initialBoatX,hullSpacing,0.6]; world.add(rightHull);

deck = sim3d.Actor(ActorName="Deck");
deck.Mobility = sim3d.utils.MobilityTypes.Movable;
deck.createShape("box",[4.5 3.2 0.25]); deck.Translation = [initialBoatX 0 1.05]; world.add(deck);

cameraPlatform = sim3d.Actor(ActorName="CameraPlatform");
cameraPlatform.Mobility = sim3d.utils.MobilityTypes.Movable;
cameraPlatform.createShape("box",[0.9 0.9 0.6]);
cameraPlatform.Translation = [initialBoatX 0 1.475]; world.add(cameraPlatform);

%% 6. FOUR CAMERAS
frontCam = sim3d.sensors.Camera(ActorName="FrontCamera",FocalLength=cameraFocalLength);
frontCam.Mobility = sim3d.utils.MobilityTypes.Movable;
frontCam.Translation = [initialBoatX+2.5 0 cameraHeight]; frontCam.Rotation = [0 0 0]; world.add(frontCam);
rightCam = sim3d.sensors.Camera(ActorName="RightCamera",FocalLength=cameraFocalLength);
rightCam.Mobility = sim3d.utils.MobilityTypes.Movable;
rightCam.Translation = [initialBoatX 1.5 cameraHeight]; rightCam.Rotation = [0 0 90]; world.add(rightCam);
rearCam = sim3d.sensors.Camera(ActorName="RearCamera",FocalLength=cameraFocalLength);
rearCam.Mobility = sim3d.utils.MobilityTypes.Movable;
rearCam.Translation = [initialBoatX-2.5 0 cameraHeight]; rearCam.Rotation = [0 0 180]; world.add(rearCam);
leftCam = sim3d.sensors.Camera(ActorName="LeftCamera",FocalLength=cameraFocalLength);
leftCam.Mobility = sim3d.utils.MobilityTypes.Movable;
leftCam.Translation = [initialBoatX -1.5 cameraHeight]; leftCam.Rotation = [0 0 270]; world.add(leftCam);

%% 7. 3D WATER SURFACE AND SHORE GEOMETRY
% Broad flat surface under the scene. If default actor material colours are
% similar in your installation, the 2D aerial plot above still clearly
% distinguishes water from land.
water = sim3d.Actor(ActorName="WaterSurface");
water.createShape("box",[200 100 0.1]); water.Translation = [20 -25 -0.08];
try, water.Color = [0.10 0.45 0.75]; catch, end
world.add(water);
shore = sim3d.Actor(ActorName="ShoreLand");
shore.createShape("box",[200 90 0.2]); shore.Translation = [20 50 0.0];
try, shore.Color = [0.35 0.60 0.25]; catch, end
world.add(shore);
% A narrow raised strip emphasizes the boundary in the 3D world.
shoreEdge = sim3d.Actor(ActorName="ShoreBoundary");
shoreEdge.createShape("box",[200 0.25 0.08]); shoreEdge.Translation = [20 shoreY 0.08];
try, shoreEdge.Color = [0.95 0.85 0.20]; catch, end
world.add(shoreEdge);

%% 8. STATIONARY OBSTACLES
obstacle1 = sim3d.Actor(ActorName="ObstacleFront"); obstacle1.createShape("box",[1.5 1.5 1.5]); obstacle1.Translation = [30 0 0.75]; world.add(obstacle1);
obstacle2 = sim3d.Actor(ActorName="ObstacleRight"); obstacle2.createShape("box",[1.5 1.5 1.5]); obstacle2.Translation = [15 6 0.75]; world.add(obstacle2);
obstacle3 = sim3d.Actor(ActorName="ObstacleLeft"); obstacle3.createShape("box",[1.5 1.5 1.5]); obstacle3.Translation = [15 -6 0.75]; world.add(obstacle3);
obstacle4 = sim3d.Actor(ActorName="ObstacleRear"); obstacle4.createShape("box",[1.5 1.5 1.5]); obstacle4.Translation = [-8 0 0.75]; world.add(obstacle4);

%% 9. RUN
fprintf('Boat speed: %.2f m/s\n',boatSpeed);
fprintf('Duration: %.1f s; expected displacement: %.1f m\n',stopTime,boatSpeed*stopTime);
fprintf('Aerial plot shoreline boundary: Y = %.1f m\n',shoreY);
disp('Starting ASV simulation...');
run(world,sampleTime,stopTime);
disp('ASV simulation completed.');

%% CALLBACKS
function outputImpl(~)
    % Intentionally empty output callback.
end

function switchAerialView(src,~)
    fig = ancestor(src,'figure');
    ax = findobj(fig,'Type','axes');
    if isempty(ax) || ~isgraphics(ax), return; end
    isAerial = getappdata(fig,'AerialMode');
    boatX = getappdata(fig,'BoatX');
    if isempty(boatX), boatX = 0; end
    if isAerial
        setappdata(fig,'AerialMode',false);
        xlim(ax,[boatX-10 boatX+18]);
        ylim(ax,[-8 12]);
        title(ax,'ASV FOLLOW VIEW — WINDOW TRACKS THE BOAT','FontWeight','bold');
        set(src,'String','Switch to Aerial Overview');
    else
        setappdata(fig,'AerialMode',true);
        xlim(ax,[-12 40]);
        ylim(ax,[-15 15]);
        title(ax,'ASV AERIAL VIEW — FIXED OVERVIEW','FontWeight','bold');
        set(src,'String','Switch to Follow View');
    end
end

function updateSimulation(world,cameraFig,cameraImages,aerialFig,aerialAx, ...
    hullLeft,hullRight,deckView,headingArrow,boatSpeed,sampleTime,stopTime,initialBoatX,shoreY,ioFolder)
    % Keep motion on simulation time and pace callbacks to avoid the boat
    % jumping through the 10-second run before the viewer can render frames.
    persistent frameCount lastFrameClock
    if isempty(frameCount)
        frameCount = 0;
        lastFrameClock = tic;
    end

    frameCount = frameCount + 1;
    elapsedTime = min((frameCount-1)*sampleTime,stopTime);
    boatX = initialBoatX + boatSpeed*elapsedTime;

    % Update all movable 3D components every simulation frame.
    world.Actors.LeftHull.Translation = [boatX -1.1 0.6];
    world.Actors.RightHull.Translation = [boatX 1.1 0.6];
    world.Actors.Deck.Translation = [boatX 0 1.05];
    world.Actors.CameraPlatform.Translation = [boatX 0 1.475];
    world.Actors.FrontCamera.Translation = [boatX+2.5 0 2.2];
    world.Actors.RightCamera.Translation = [boatX 1.5 2.2];
    world.Actors.RearCamera.Translation = [boatX-2.5 0 2.2];
    world.Actors.LeftCamera.Translation = [boatX -1.5 2.2];

    % Update the aerial diagram every frame.
    if isgraphics(aerialFig,'figure') && isgraphics(aerialAx,'axes')
        setappdata(aerialFig,'BoatX',boatX);
        if isgraphics(hullLeft), set(hullLeft,'Position',[boatX-3 -1.5 6 0.8]); end
        if isgraphics(hullRight), set(hullRight,'Position',[boatX-3 0.7 6 0.8]); end
        if isgraphics(deckView), set(deckView,'Position',[boatX-2.25 -1.6 4.5 3.2]); end
        if isgraphics(headingArrow)
            set(headingArrow,'XData',boatX,'YData',0,'UData',3,'VData',0);
        end
        if ~getappdata(aerialFig,'AerialMode')
            xlim(aerialAx,[boatX-10 boatX+18]);
            ylim(aerialAx,[-8 12]);
        end
    end

    if mod(frameCount,60)==0
        fprintf('Time: %.1f s | Boat X: %.2f m\n',elapsedTime,boatX);
    end

    % FRAME-DRIVEN CAMERA EXCHANGE:
    % Read, send, and display each camera on every simulation callback.
    % There is no mod(frameCount,6) transfer interval.
    if isgraphics(cameraFig,'figure')
        names = {'FrontCamera','RightCamera','RearCamera','LeftCamera'};
        shortNames = {'front','right','rear','left'};
        for k = 1:4
            try
                img = read(world.Actors.(names{k}));
                if isempty(img) || ~isnumeric(img) || ndims(img)~=3 || size(img,3)~=3
                    continue;
                end
                inputPath = fullfile(ioFolder,[shortNames{k} '_input.png']);
                % Unique input temp file avoids collision with Python/Windows locks.
                tempPath = fullfile(ioFolder,[shortNames{k} '_' char(java.util.UUID.randomUUID) '_input_tmp.png']);
                imwrite(img,tempPath);
                movefile(tempPath,inputPath,'f');

                overlayPath = fullfile(ioFolder,[shortNames{k} '_overlay.png']);
                if isfile(overlayPath)
                    overlay = imread(overlayPath);
                    if ~isempty(overlay) && ndims(overlay)==3 && size(overlay,3)==3
                        img = overlay;
                    end
                end
                if isgraphics(cameraImages(k),'image')
                    set(cameraImages(k),'CData',img);
                end
            catch ME
                if mod(frameCount,120)==0
                    fprintf('Camera %s transfer warning: %s\n',shortNames{k},ME.message);
                end
            end
        end
        drawnow limitrate;
    end

    % Keep the UI responsive after each callback. Frame exchange is driven
    % by simulation updates; Python processes the newest available frame.
    drawnow limitrate;
end
